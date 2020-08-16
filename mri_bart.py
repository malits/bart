#!/usr/bin/python3
#%matplotlib qt

import logging

import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import ctypes
import os
import sys

scriptdir = os.path.dirname(os.path.realpath(__file__))
os.environ['TOOLBOX_PATH']=os.path.join(scriptdir, 'bart')


sys.path.insert(0, os.path.join(scriptdir, 'bart/python'))
import cfl



sys.path.insert(0, os.path.join(scriptdir, 'itreg'))
import regpy as rp
import regpy.operators.mri as rpm
import regpy.solvers.irgnm as rpirgnm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)


bartso = ctypes.CDLL(os.path.join(scriptdir, 'bart.so'));


class bart_iovec(ctypes.Structure):
	_fields_ = [("N", ctypes.c_uint),
			("dims", ctypes.POINTER(ctypes.c_long)),
			("strs", ctypes.POINTER(ctypes.c_long)),
			("size", ctypes.c_size_t)]

bartso.nlop_apply.restype = None;
bartso.nlop_derivative.restype = None;
bartso.nlop_adjoint.restype = None;
bartso.nlop_free.restype = None;
bartso.nlop_codomain.restype = ctypes.POINTER(bart_iovec)
bartso.nlop_domain.restype = ctypes.POINTER(bart_iovec)



class nlop:
	class bart_nlop(ctypes.Structure):
		pass

	def __init__(self, cnlop: bart_nlop):
		self.cnlop = cnlop
		self.codomain = bartso.nlop_codomain(cnlop).contents
		self.domain = bartso.nlop_domain(cnlop).contents
		self.oshape = ctypes.cast(self.codomain.dims, ctypes.POINTER(self.codomain.N * ctypes.c_long)).contents
		self.ishape = ctypes.cast(self.domain.dims, ctypes.POINTER(self.domain.N * ctypes.c_long)).contents

	def __del__(self):
		bartso.nlop_free(self.cnlop)

	def apply(self, src: np.array):
		dst = np.asfortranarray(np.empty(self.oshape, dtype=np.complex64))
		assert self.codomain.size == dst.itemsize
		src = src.astype(np.complex64, copy=False)
		bartso.nlop_apply(self.cnlop,
			dst.ndim, dst.ctypes.shape_as(ctypes.c_long), dst.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
			src.ndim, src.ctypes.shape_as(ctypes.c_long), src.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
		return dst
	
	
	def derivative(self, src: np.array):
		dst = np.asfortranarray(np.empty(self.oshape, dtype=np.complex64))
		assert self.codomain.size == dst.itemsize
		src = src.astype(np.complex64, copy=False)
		bartso.nlop_derivative(self.cnlop,
			dst.ndim, dst.ctypes.shape_as(ctypes.c_long), dst.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
			src.ndim, src.ctypes.shape_as(ctypes.c_long), src.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
		return dst
	
	def adjoint(self, src: np.array):
		dst = np.asfortranarray(np.empty(self.ishape, dtype=np.complex64))
		assert self.domain.size == dst.itemsize
		src = src.astype(np.complex64, copy=False)
		bartso.nlop_adjoint(self.cnlop,
			dst.ndim, dst.ctypes.shape_as(ctypes.c_long), dst.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
			src.ndim, src.ctypes.shape_as(ctypes.c_long), src.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
		return dst



class noir_model_conf_s(ctypes.Structure):
	pass

class bart_linop(ctypes.Structure):
	pass

class bart_noir_op_s(ctypes.Structure):
	pass


class noir_s(ctypes.Structure):
	_fields_ = [ ("nlop", ctypes.POINTER(nlop.bart_nlop)),
		     ("linop", ctypes.POINTER(bart_linop)),
	 	    ("noir_op", ctypes.POINTER(bart_noir_op_s)) ]


class BartNoir(rp.operators.Operator):
	"""Operator that implements the multiplication between density and coil profiles. The domain
	is a direct sum of the `grid` (for the densitiy) and a `regpy.discrs.UniformGrid` of `ncoils`
	copies of `grid`, stacked along the 0th dimension.

	Parameters
	----------
	grid : regpy.discrs.UniformGrid
		The grid on which the density is defined.
	ncoils : int
		The number of coils.
		"""

	def __init__(self, grid, ncoils, psf):
		assert isinstance(grid, rp.discrs.UniformGrid)
		assert grid.ndim == 2
		self.grid = grid
		"""The density grid."""
		self.coilgrid = rp.discrs.UniformGrid(ncoils, *grid.axes, dtype=grid.dtype)
		"""The coil grid, a stack of copies of `grid`."""
		self.ncoils = ncoils
		"""The number of coils."""
		super().__init__(
			domain = self.grid + self.coilgrid,
			codomain = rp.discrs.Discretization(np.count_nonzero(psf)*ncoils, dtype=self.grid.dtype)
			)

		sd = list(self.grid.shape)
		sd += [1] * (3 - len(sd))

		shaped = [1]*16
		shaped[:3] = sd

		shapec = shaped.copy()
		shapec[3] = ncoils

		self.dimsd = (ctypes.c_long * len(shaped))( *shaped)
		self.dimsc = (ctypes.c_long * len(shaped))( *shapec)

		self.psff = psf.astype(np.complex64)

		psfnz = np.zeros(self.coilgrid.shape, dtype=bool)
		for c in range(ncoils):
			psfnz[c,:,:] = psf

		self.psfnz = psfnz.flatten()
		
		bartso.noir_create.restype = noir_s;
		self.nconf = noir_model_conf_s.in_dll(bartso, "noir_model_conf_defaults");
		self.ns = bartso.noir_create(self.dimsc, None, self.psff.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)), ctypes.pointer(self.nconf))
		self.nl = nlop(self.ns.nlop)
		
	def _bartpreproc(self, x):
		density, coils = self.domain.split(x)
		d = np.asfortranarray(np.reshape(density.transpose(), self.dimsd))
		c = np.asfortranarray(np.reshape(coils.transpose(), self.dimsc))
		x_b = np.asfortranarray(np.concatenate((d, c), 3))
		return x_b
	
	
	
	def _bartpostproc(self, d_b):
		d = np.transpose(d_b).flatten()
		return d[self.psfnz]

	def _eval(self, x, differentiate=False):
		density, coils = self.domain.split(x)

		x_b = self._bartpreproc(x)
		dst = self.nl.apply(x_b)

		return self._bartpostproc(dst)

	def _derivative(self, x):
		x2 = self._bartpreproc(x)
		dst = self.nl.derivative(x2)
		dst2 = self._bartpostproc(dst)
		return dst2

	def _adjoint(self, y):
		y_tmp = self.coilgrid.zeros().flatten()
		y_tmp[self.psfnz] = y.flatten()
		y_b = np.asfortranarray(y_tmp.transpose()).reshape(self.dimsc)
		
		x = self.nl.adjoint(y_b)
		d = x[:,:,:,0:1,...]
		c = x[:,:,:,1:,...]

		dst = self.domain.zeros()
		dd, dc = self.domain.split(dst)
		dd[...] = np.ascontiguousarray(d.transpose().squeeze())
		dc[...] = np.ascontiguousarray(c.transpose().squeeze())

		return dst


	def _forward_coils(self, x):
		xc = x.copy()
		density, coils = self.domain.split(xc)
		
		c = np.asfortranarray(np.reshape(np.transpose(coils),
								   self.dimsc)).astype(np.complex64, copy=False)

		bartso.noir_forw_coils.restype = None;
		bartso.noir_forw_coils(self.ns.linop, c.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
						 c.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
		coils[...] = np.ascontiguousarray(c.transpose().squeeze())

		return xc

	def __repr__(self):
		return rp.util.make_repr(self, self.grid, self.ncoils)

def plot_nl(x, d, ni):
	nr = int(np.trunc(np.sqrt(ni)))
	f, ax = plt.subplots(nrows=nr, ncols=ni//nr, constrained_layout=True)
	im,c = d.split(x)
	print(ax.shape)
	for i, ax in enumerate(ax.flat):
		if i == 0:
			ax.imshow(np.abs(im))
			ax.set_title("Image")
		else:
			ax.imshow(np.abs(c[i-1,...]))
			ax.set_title("Coil {!s}".format(i-1))
			
def plot_ci(x):
	nr = int(np.trunc(np.sqrt(nc)))
	f, ax = plt.subplots(nrows=nr, ncols=nc//nr, constrained_layout=True)
	for i, ax in enumerate(ax.flat):
		ax.imshow(np.abs(x[i,...]))
		ax.set_title("Coil {!s}".format(i))


#%% Read and preprocess data

sobolev_index = 32
a = 220.
noiselevel = None

data_pre = scriptdir
# datafile = os.path.join(scriptdir, 'data/unders_2')
datafile= os.path.join(scriptdir, 'data/unders_2_v8')
# datafile= os.path.join(scriptdir, 'data/unders_4')



exact_data_b = cfl.readcfl(datafile)
exact_data = np.ascontiguousarray(np.transpose(exact_data_b)).squeeze()


bart_reference = cfl.readcfl(datafile + '_bartref')



X = exact_data_b.shape[1]
ncoils = exact_data_b.shape[3]

grid = rp.discrs.UniformGrid((-1, 1, X), (-1, 1, X), dtype=np.complex64)

pattern = rpm.estimate_sampling_pattern(exact_data)

#%% Reconstruction

bartop = BartNoir(grid, ncoils, pattern)

setting = rp.solvers.HilbertSpaceSetting(op=bartop, Hdomain=rp.hilbert.L2, Hcodomain=rp.hilbert.L2)

exact_data_itreg = exact_data[:,pattern].flatten()
exact_data_itreg = exact_data_itreg/setting.Hcodomain.norm(exact_data_itreg)*100

if noiselevel is not None:
	data = (exact_data_itreg + noiselevel * bartop.codomain.randn(dtype=complex)).astype(np.complex64)
else:
	data = exact_data_itreg

init = bartop.domain.zeros()
init_density, init_coils = bartop.domain.split(init)
init_density[...] = 1.
init_coils[...] = 0.

solver = rpirgnm.IrgnmCG(
    setting=setting,
    data=data,
    regpar=1,
    regpar_step=1./2.,
    init=init,
    cgstop=5
)

stoprule = (
    rp.stoprules.CountIterations(max_iterations=11) +
    rp.stoprules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(exact_data_itreg - data),
        tau=0.5
    )
)

# Plotting setup
plt.ion()
#fig, axes = plt.subplots(ncols=3, constrained_layout=True)
fig, axes = plt.subplots(ncols=3)
# bars = [mp.colorbar.make_axes(ax)[0] for ax in axes]

axes[0].set_title('reference solution')
axes[1].set_title('reconstruction')
axes[2].set_title('difference x10')

# Plot exact solution
ref = axes[0].imshow(np.fliplr(np.abs(bart_reference.transpose().squeeze()).transpose()), origin='lower')
ref.set_clim((0, ref.get_clim()[1]))

# Run the solver, plot iterates
for reco, reco_data in solver.until(stoprule):
    reco_postproc = rpm.normalize(*bartop.domain.split(bartop._forward_coils(reco)))
    im = axes[1].imshow(np.fliplr(np.abs(reco_postproc).transpose()), origin='lower')
    im.set_clim(ref.get_clim())
    diff = axes[2].imshow(np.fliplr((np.abs(bart_reference.transpose().squeeze()) - np.abs(reco_postproc)).transpose()), origin='lower')
    diff.set_clim((0, ref.get_clim()[1]/10.))
    plt.pause(0.5)



plt.ioff()
plt.show()


#%% Test equivalence of operators

print('Test equivalence of bart and itreg operator')

bartop = BartNoir(grid, ncoils, pattern)

full_mri_op = rpm.parallel_mri(grid=grid, ncoils=ncoils, centered=True)
sampling = rpm.cartesian_sampling(full_mri_op.codomain, mask=pattern)
mri_op = sampling * full_mri_op
smoother = rpm.sobolev_smoother(mri_op.domain, sobolev_index, factor=a, centered=True)
smoothed_op = mri_op * smoother

test_input = bartop.domain.zeros()
test_density, test_coils = bartop.domain.split(test_input)
test_density[...] = bartop.grid.randn(dtype=np.complex).astype(np.complex64)
test_coils[...] = bartop.coilgrid.randn(dtype=np.complex).astype(np.complex64)

bartfor = bartop(test_input)
itfor = smoothed_op(test_input).astype(np.complex64)
print('Forward allclose?: ', np.allclose(bartfor, itfor, rtol=1e-4, atol=1e-6))

bartopder = rp.operators.Derivative(bartop)
litop, litopder = smoothed_op.linearize(test_input)
bartder = bartopder(test_input)
itder = litopder(test_input)
print('Derivative allclose?: ', np.allclose(bartder, itder, rtol=1e-4, atol=1e-6))

test_input_adj= bartop.codomain.randn(dtype=np.complex).astype(np.complex64)

bartopadj = rp.operators.Adjoint(bartopder)
bartadj = bartopadj(test_input_adj)
litop, litopder = smoothed_op.linearize(test_input)
itadj = litopder.adjoint(test_input_adj)
print('Adj allclose?: ', np.allclose(bartadj, itadj, rtol=1e-4, atol=1e-6))


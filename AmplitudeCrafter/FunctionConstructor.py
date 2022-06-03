from jitter.amplitudes.dalitz_plot_function import DalitzDecay, chain
from jitter.constants import spin as sp
from AmplitudeCrafter.Resonances import map_arguments
from jax import numpy as jnp

def run_lineshape(resonance_tuple,args,mapping_dict):
    s,p,hel,lineshape,M0,d,p0 = resonance_tuple
    lineshape = lineshape(*map_arguments(args,mapping_dict))
    return (s,p,hel,lineshape,M0,d,p0)

def construct_function(masses,spins,parities,param_names,params,mapping_dict,resonances,resonance_tuples,bls_in,bls_out,resonance_args,smp,phsp):

    resonances_filled = [run_lineshape(r,resonance_args[i][j],mapping_dict) for i, res in enumerate(resonance_tuples) for j,r in enumerate(res) ]
    free_indices = [not r.fixed() for res in resonances for r in res ]

    decay = DalitzDecay(*masses,*spins,*parities,smp,resonances_filled,bls_in,bls_out,phsp=phsp)

    start = map_arguments(params,mapping_dict)

    def fill_args(args,mapping_dict):
        for name, val in zip(param_names,args):
            mapping_dict[name] = val
        return mapping_dict

    def update():
        for i,l in enumerate(free_indices):
            for j, free in enumerate(l):
                if free:
                    resonances_filled[i][j] = run_lineshape(resonance_tuples[i][j],resonance_args[i][j],mapping_dict)

    def f(args):
        fill_args(args,mapping_dict)
        update()
        bls_in_mapped = map_arguments(bls_in,mapping_dict)
        bls_out_mapped = map_arguments(bls_out,mapping_dict)
        def O(nu,lambdas):
            tmp = chain(decay,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3) + chain(decay,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2) + chain(decay,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1)
            return tmp

        ampl = sum(sum(jnp.abs(O(ld,[la,0,0]))**2  for la in sp.direction_options(decay["sa"])) for ld in sp.direction_options(decay["sa"]))
        return ampl

    return f


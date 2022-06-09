from jitter.amplitudes.dalitz_plot_function import DalitzDecay, chain
from jitter.constants import spin as sp
from numpy import ma
from AmplitudeCrafter.Resonances import map_arguments, needed_parameter_names
from jax import numpy as jnp
from jitter.amplitudes.dalitz_plot_function import helicity_options
from jax import jit

def run_lineshape(resonance_tuple,args,mapping_dict):
    s,p,hel,lineshape,M0,d,p0 = resonance_tuple
    lineshape = lineshape(*map_arguments(args,mapping_dict))
    return (s,p,hel,lineshape,M0,d,p0)



def construct_function(masses,spins,parities,param_names,params,mapping_dict,resonances,resonance_tuples,bls_in,bls_out,resonance_args,smp,phsp,total_absolute=True):
    mapping_dict_global = mapping_dict
    resonances_filled = [[run_lineshape(r,resonance_args[i][j],mapping_dict) for j,r in enumerate(res)] for i, res in enumerate(resonance_tuples)  ]
    free_indices = [[not r.fixed() for r in res ] for res in resonances ]
    bls_in_mapped = map_arguments(bls_in,mapping_dict)
    bls_out_mapped = map_arguments(bls_out,mapping_dict)

    decay = DalitzDecay(*masses,*spins,*parities,smp,resonances_filled,[bls_in_mapped,bls_out_mapped],phsp=phsp)
    
    needed_param_names = needed_parameter_names(param_names)
    # we need to translate all _complex values into real and imaginary
    start = map_arguments(needed_param_names,mapping_dict)

    def fill_args(args,mapping_dict):
        if len(args) == 1:
            # wierd bug...
            # need to investigate this
            args = args[0]
        dtc = mapping_dict.copy()
        for name, val in zip(needed_param_names,args):
            dtc[name] = val
        return dtc

    def update(mapping_dict):
        for i,l in enumerate(free_indices):
            for j, free in enumerate(l):
                if free:
                    resonances_filled[i][j] = run_lineshape(resonance_tuples[i][j],resonance_args[i][j],mapping_dict)
    if total_absolute:
        @jit
        def f(args):
            mapping_dict = fill_args(args,mapping_dict_global)
            bls_in_mapped = map_arguments(bls_in,mapping_dict)
            bls_out_mapped = map_arguments(bls_out,mapping_dict)
            update(mapping_dict)

            def O(nu,lambdas):       
                tmp = chain(decay,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3) + chain(decay,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2) + chain(decay,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1)
                return tmp
            ampl =            sum(
                sum(
                    jnp.abs(O(ld,[la,lb,lc]))**2  
                        for la,lb,lc in helicity_options(decay["sa"])
                            ) for ld in sp.direction_options(decay["sd"]))
            return ampl
    else:
        @jit
        def f(args,nu,lambdas):
            mapping_dict = fill_args(args,mapping_dict_global)
            bls_in_mapped = map_arguments(bls_in,mapping_dict)
            bls_out_mapped = map_arguments(bls_out,mapping_dict)
            update(mapping_dict)

            def O(nu,lambdas):       
                tmp = chain(decay,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3) + chain(decay,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2) + chain(decay,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1)
                return tmp
            return O(nu,lambdas)

    return f, start


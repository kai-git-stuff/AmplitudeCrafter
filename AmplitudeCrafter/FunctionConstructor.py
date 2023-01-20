from jitter.amplitudes.dalitz_plot_function import DalitzDecay, chain
from jitter.constants import spin as sp
from numpy import ma
from AmplitudeCrafter.Resonances import map_arguments
from jax import numpy as jnp
from jitter.amplitudes.dalitz_plot_function import helicity_options
from jax import jit
from jitter.kinematics import two_body_momentum
from jitter.dynamics import orbital_barrier_factor

def run_lineshape(resonance_tuple,args,mapping_dict,bls_in,bls_out):
    s,p,hel,lineshape_func,_,_,_ = resonance_tuple
    lineshape = {}
    for LS,b in bls_out.items():
        for LS_i,b_i in bls_in.items():
            L,S = LS
            L_0, S_0 = LS_i
            mapping_dict["L"].update(L) # set the correct angular momentum
            mapping_dict["L_0"].update(L_0) # set the correct angular momentum
            lineshape[(L_0,L)] = lineshape_func(*map_arguments(args))

    return (s,p,hel,lineshape,None,None,None)

def construct_function(masses,spins,parities,params,mapping_dict,resonances,resonance_tuples,bls_in,bls_out,resonance_args,smp,phsp,total_absolute=True,just_in_time_compile=True, numericArgs=True):
    mapping_dict_global = mapping_dict
    free_indices = [[not r.fixed() for r in res ] for res in resonances ]
    bls_in_mapped = map_arguments(bls_in)
    bls_out_mapped = map_arguments(bls_out)
    resonances_filled = [[run_lineshape(r,resonance_args[i][j],mapping_dict,bls_in_mapped[i][j],bls_out_mapped[i][j]) for j,r in enumerate(res)] for i, res in enumerate(resonance_tuples)  ]

    decay = DalitzDecay(*masses,*spins,*parities,smp,resonances_filled,[bls_in_mapped,bls_out_mapped],phsp=phsp)
    
    needed_params = [p for p in params if not p.const]
    # we need to translate all _complex values into real and imaginary
    print(needed_params)
    start = map_arguments(needed_params,numeric=numericArgs)

    def fill_args(args,mapping_dict):
        if len(args) == 1:
            # wierd bug...
            # need to investigate this
            args = args[0]
        dtc = mapping_dict
        for p, val in zip(needed_params,args):
            p.update(val)
        return dtc

    def update(mapping_dict,bls_out):
        for i,l in enumerate(free_indices):
            for j, free in enumerate(l):
                if free:
                    resonances_filled[i][j] = run_lineshape(resonance_tuples[i][j],resonance_args[i][j],mapping_dict,bls_in[i][j],bls_out[i][j])

    if total_absolute:
        def f(args):
            mapping_dict = fill_args(args,mapping_dict_global)
            bls_in_mapped = map_arguments(bls_in)
            bls_out_mapped = map_arguments(bls_out)
            update(mapping_dict,bls_out_mapped)

            def O(nu,lambdas):       
                tmp = chain(decay,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3) + chain(decay,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2) + chain(decay,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1)
                return tmp
            ampl =  sum(
                sum(
                    jnp.abs(O(ld,[la,lb,lc]))**2  
                        for la,lb,lc in decay["HelicityOptions"]
                            ) for ld in sp.direction_options(decay["sd"]))
            return ampl
    else:
        def f(args,nu,*lambdas):
            mapping_dict = fill_args(args,mapping_dict_global)
            bls_in_mapped = map_arguments(bls_in)
            bls_out_mapped = map_arguments(bls_out)
            update(mapping_dict,bls_out_mapped)

            def O(nu,lambdas):       
                tmp = chain(decay,nu,*lambdas,resonances_filled[2],bls_in_mapped[2],bls_out_mapped[2],3) + chain(decay,nu,*lambdas,resonances_filled[1],bls_in_mapped[1],bls_out_mapped[1],2) + chain(decay,nu,*lambdas,resonances_filled[0],bls_in_mapped[0],bls_out_mapped[0],1)
                return tmp
            return O(nu,lambdas)
    if just_in_time_compile:
        f = jit(f)
    return f, start
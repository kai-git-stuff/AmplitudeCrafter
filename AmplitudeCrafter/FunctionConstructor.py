from jitter.amplitudes.dalitz_plot_function_no_buffer import ThreeBodyAmplitude
from jitter.constants import spin as sp
from AmplitudeCrafter.Resonances import map_arguments
from jax import numpy as jnp
from jax import jit
from jitter.fitting import FitParameter
from decayangle.kinematics import mass_squared

def run_lineshape(resonance_tuple,args,mapping_dict,bls_in,bls_out,masses):
    """
    Performs the execution of a single Resonance lineshape
    L and L_0 values are set manually
    The resonance tuple includes information no longer needed
    For legacy resoans it is still carried, but will be droppend in future versions
    """
    s,p,hel,lineshape_func,channel = resonance_tuple
    lineshape = {}
    i,j,k = {1:[2,3,1], 2:[3,1,2], 3:[1,2,3]}[channel]
    mapping_dict["M0"] = masses[0]
    mapping_dict["MI"] = masses[i]
    mapping_dict["MJ"] = masses[j]
    mapping_dict["MK"] = masses[k]
    new_bls_out = {}
    for LS,b in bls_out.items():
        for LS_i,b_i in bls_in.items():
            L,S = LS
            L_0, S_0 = LS_i
            mapping_dict["L"] = L  # set the correct angular momentum
            mapping_dict["L_0"] = L_0 # set the correct angular momentum
            lineshape[(L_0,L)] = lineshape_func(*map_arguments(args,mapping_dict=mapping_dict))
            new_bls_out[LS] = lineshape[(L_0,L)] * b

    return new_bls_out

def construct_function(masses,spins,params,mapping_dict,
                        resonances, resonance_tuples,bls_in,bls_out,resonance_args,momenta,
                        numericArgs=True):
    """
    Function to construct the actual ampltude function from the Dalitz Amplitude
    The function will take all non fixed parameters as defined in the yml files

    total_absolute will descide weather the dalitz plot funciton will be returned or
    the sum over all helicity configurations
    """
    
    needed_params = params
    needed_names = [p.name for p in needed_params] # TODO: Names are wrong here!!! They do not reflect the names in the mapping dict!
    free_indices = {i:[not r.fixed() for r in resonances[i] ] for i in resonances }
    bls_in_mapped = map_arguments(bls_in,mapping_dict=mapping_dict)
    bls_out_mapped = map_arguments(bls_out,mapping_dict=mapping_dict)
    # resonances_filled = [
    #     [
    #         run_lineshape(r,
    #                       resonance_args[i][j],
    #                       mapping_dict,
    #                       bls_in_mapped[i][j],
    #                       bls_out_mapped[i][j],
    #                       masses) 
    #                       for j,r in enumerate(res)
    #     ] 
    #         for i, res in enumerate(resonance_tuples)  
    #                     ]
    mapping_dict_global = mapping_dict
    decay = ThreeBodyAmplitude(*spins, resonances, momenta)

    # we need to translate all _complex values into real and imaginary
    start = map_arguments(needed_params,numeric=numericArgs)

    def fill_args(args,mapping_dict):
        """
        Put the arguments into the correct places in the mapping dict
        This has to be used, so no JAX Tracer objects leak
        Thus the parameter.update() functionlaity may not be used!
        Instead the parmeter value can be retrieved by using parameter(value_dict=mapping_dict)
        This will read the needed value from the dict, but still make use of the internal logic to construct
        complex numbers and adhere to naming schemes.
        """
        if len(args) == 1:
            # wierd bug...
            # need to investigate this
            args = args[0]
        dtc = mapping_dict.copy()
        for param_name, val in zip(needed_names,args):
            if isinstance(val,FitParameter):
                val = val()
            dtc[param_name] = val
        return dtc

    def update(mapping_dict,bls_out):
        """
        We only need to update the resonance lineshapes, where parameters of the lineshape are free.
        we will then apply the lineshape to the bls_out, since this is 
        """
        bls_out_new = []
        print(bls_out)
        for i,k in enumerate(free_indices):
            bls_out_new.append([])
            for j, free in enumerate(free_indices[k]):
                bls_out_new[i].append(bls_out[i][j])
                bls_out_with_lineshape = run_lineshape(resonance_tuples[i][j],resonance_args[i][j],mapping_dict,bls_in[i][j],bls_out[i][j],masses)
                bls_out_new[i][j] = bls_out_with_lineshape
        return bls_out_new

    def f(args,nu,*lambdas):
        # The resonance will be put with the bls couplings, since the form of the resonance is depending on the L value
        mapping_dict = fill_args(args,mapping_dict_global)
        bls_in_mapped = map_arguments(bls_in,mapping_dict=mapping_dict)
        bls_out_mapped = map_arguments(bls_out,mapping_dict=mapping_dict)
        bls_out_mapped_with_lineshape = update(mapping_dict,bls_out_mapped)

        def O(nu,lambdas):       
            tmp = decay(nu,*lambdas,bls_in_mapped,bls_out_mapped_with_lineshape)
            return tmp
        return O(nu,lambdas)
    return f, start
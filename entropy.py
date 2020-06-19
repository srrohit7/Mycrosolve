import numpy as np
from ase import units
from scipy.special import jn, i0, iv

def rotationalinertia(atoms, indices=slice(None, None)):
    """Calculates the three principle moments of inertia for an ASE atoms
    object. This uses the atomic masses from ASE, which (if not explicitly
    specified by the user) gives an inexact approximation of an isotopically
    averaged result. Units are in amu*angstroms**2."""

    atoms = atoms[indices]

    # Calculate the center of mass.
    xcm, ycm, zcm = atoms.get_center_of_mass()
    masses = atoms.get_masses()

    # Calculate moments of inertia in the current frame of reference.
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = 0., 0., 0., 0., 0., 0.
    for index, atom in enumerate(atoms):
        m = masses[index]
        x = atom.x - xcm
        y = atom.y - ycm
        z = atom.z - zcm
        Ixx += m * (y**2. + z**2.)
        Iyy += m * (x**2. + z**2.)
        Izz += m * (x**2. + y**2.)
        Ixy += m * x * y
        Ixz += m * x * z
        Iyz += m * y * z
    # Create the inertia tensor in the current frame of reference.
    I_ = np.matrix([[ Ixx, -Ixy, -Ixz],
                    [-Ixy,  Iyy, -Iyz],
                    [-Ixz, -Iyz,  Izz]])
    # Find the eigenvalues, which are the principle moments of inertia.
    I = np.linalg.eigvals(I_)
    return I


def campbellsellers_entropy(item, temperature, pressure):
    """
    The CS entropy method is quite unique from the others. It is an empirical relationship.
    It is calculated by assuming an ideal gas and then removing 2/3rds of the entropy.
    This approximation has been put forward by Campbell and Sellers.
    Therefore, when making the species file the used should be reminded that the need the following data:
    GAS PHASE POSCAR
    GAS PHASE frequencies
    ADSORBATE frequencies

    the gas phase POSCAR is unneccesary. I recommend we have this as a special exception in the species and reaction code
    """
    # apply the Campbell Sellers approximation
    eV_Jmol = 1.60218e-19 * units._Nav # convert atomic entropy in eV/K to J/(mol.K)

    # Campbell Sellers in SI units as derived in the original paper
    S = ( ( 0.70 * eV_Jmol * idealgas_entropy(item.cs_gas_ref, temperature, pressure) ) - ( 3.3 * units._k * units._Nav ) ) * ( 1. / eV_Jmol )
    return S

def harmonic_entropy(item, temperature, pressure):
    vibrational_energies = item.vibrational_energies
    vibrational_frequencies = item.vibrational_frequencies
    vibrational_imaginary = item.vibrational_imaginary

    # We need to check if the frequency is imaginary. The reaction/species file should denote which frequencies had the string f/i
    for x, energy in enumerate(vibrational_energies):
        if vibrational_imaginary[x] or vibrational_frequencies[x] < 20:
            vibrational_energies[x] = 2.47968e-03
            vibrational_frequencies[x] = 20

    # Once these values are re-scaled the calculation is identical to the IG case
    S = 0

    kT = units.kB * temperature
    S_v = 0.
    for energy in vibrational_energies:
        x = energy / kT
        S_v +=  - np.log(1-np.exp(-x))
    S_v *= units.kB

    S += S_v

    return S

def idealgas_entropy(item, temperature, pressure):
    vibrational_energies = item.vibrational_energies
    vibrational_frequencies = item.vibrational_frequencies

    if item.geometry == 'Nonlinear':
        vibrational_energies = item.vibrational_energies[:item.entropy_frequencies-6]
        vibrational_frequencies = item.vibrational_frequencies[:item.entropy_frequencies-6]
        vibrational_imaginary = item.vibrational_imaginary[:item.entropy_frequencies-6]
    elif item.geometry == 'Linear':
        vibrational_energies = item.vibrational_energies[:item.entropy_frequencies-5]
        vibrational_frequencies = item.vibrational_frequencies[:item.entropy_frequencies-5]
        vibrational_imaginary = item.vibrational_imaginary[:item.entropy_frequencies-5]
    elif item.geometry == 'Monatomic':
        item.vibrational_energies = []
        vibrational_frequencies = []
        vibrational_imaginary = []

    S = 0.0

    # Translational entropy (term inside the log is in SI units).
    mass = sum(item.atoms.get_masses()) * units._amu  # kg/molecule
    S_t = (2 * np.pi * mass * units._k *
           temperature / units._hplanck**2)**(3.0 / 2)
    S_t *= units._k * temperature / pressure
    S_t = units.kB * (np.log(S_t) )

    S += S_t


    # Rotational entropy (term inside the log is in SI units).
    # We need to check if the frequency is imaginary. The reaction/species file should denote which frequencies had the string f/i
    if item.geometry == 'Monatomic':
        S_r = 0.0
    elif item.geometry == 'Nonlinear':
        inertias = (rotationalinertia(item.atoms) * units._amu /
                    (10.0**10)**2)  # kg m^2
        S_r = np.sqrt(np.pi * np.product(inertias)) / item.sigma
        S_r *= (8.0 * np.pi**2 * units._k * temperature /
                units._hplanck**2)**(3.0 / 2.0)
        S_r = units.kB * (np.log(S_r) )
    elif item.geometry == 'Linear':
        inertias = (rotationalinertia(item.atoms) * units._amu /
                (10.0**10)**2)  # kg m^2
        inertia = max(inertias)  # should be two identical and one zero
        S_r = (8 * np.pi**2 * inertia * units._k * temperature /
                item.sigma / units._hplanck**2)
        S_r = units.kB * (np.log(S_r) )
    S += S_r

    # Vibrational entropy
    for x, energy in enumerate(vibrational_energies):
        if vibrational_imaginary[x] or vibrational_frequencies[x] < 20:
            vibrational_energies[x] = 2.47968e-03
            vibrational_frequencies[x] = 20

    kT = units.kB * temperature
    S_v = 0.
    for energy in vibrational_energies:
        x = energy / kT
        S_v += - np.log(1-np.exp(-x))
    S_v *= units.kB
    S += S_v

    return S

def rigidbody_entropy(item, temperature, pressure):
    if item.geometry == 'Nonlinear':
        vibrational_energies = item.vibrational_energies[:item.entropy_frequencies-2]
        vibrational_frequencies = item.vibrational_frequencies[:item.entropy_frequencies-2]
        vibrational_imaginary = item.vibrational_imaginary[:item.entropy_frequencies-2]
    elif item.geometry == 'Linear':
        vibrational_energies = item.vibrational_energies[:item.entropy_frequencies-1]
        vibrational_frequencies = item.vibrational_frequencies[:item.entropy_frequencies-1]
        vibrational_imaginary = item.vibrational_imaginary[:item.entropy_frequencies-1]

    # Once these values are re-scaled the calculation is identical to the IG case
    for x, energy in enumerate(vibrational_energies):
        if vibrational_imaginary[x] or vibrational_frequencies[x] < 20:
            vibrational_energies[x] = 2.47968e-03
            vibrational_frequencies[x] = 20

    S = 0.0

    if item.geometry == 'Nonlinear':
        inertias = (rotationalinertia(item.atoms) * units._amu /
                    (10.0**10)**2)  # kg m^2
        S_r = np.sqrt(np.pi * inertias[0] * inertias[1]) / item.sigma
        S_r *= (8.0 * np.pi**2 * units._k * temperature /
                units._hplanck**2)**(2.0 / 2.0)
        S_r = units.kB * (np.log(S_r) )
    elif item.geometry == 'Linear':
        inertias = (rotationalinertia(item.atoms) * units._amu /
                (10.0**10)**2)  # kg m^2
        inertia = max(inertias)  # should be two identical and one zero
        S_r = (8 * np.pi**2 * inertia * units._k * temperature /
                item.sigma / units._hplanck**2)**(1.0/ 2.0)
        S_r = units.kB * (np.log(S_r) )
    S += S_r

    kT = units.kB * temperature

    S_v = 0.

    for energy in vibrational_energies:
        x = energy / kT
        S_v += - np.log(1-np.exp(-x))
    S_v *= units.kB

    S += S_v

    return S

def advanced_entropy(item, temperature, pressure):
    S_tot = 0

    inertias = (rotationalinertia(item.atoms) * units._amu /
                (10.0**10)**2)  # kg m^2

    ad_args = item.entropy_args.get("AD", {})
    none_list = [None] * item.entropy_frequencies
    ad_partition_function = ad_args.get("partition_function", none_list)
    ad_partial_mass = ad_args.get("partial_mass", none_list)
    ad_barrier = ad_args.get("barrier", none_list)
    ad_moments = ad_args.get("moment", none_list)
    ad_diffusion = ad_args.get("diffusion_length", none_list)
    ad_symmetry = ad_args.get("symmetry", none_list)


    # We go through each degree of freedom provided in the item file
    for index in range(item.entropy_frequencies):
        freq = item.vibrational_frequencies[index]
        energy = item.vibrational_energies[index]
        try:
            dof_type = ad_partition_function[index].lower() # Lowercase for easy matching
        except:
            dof_type = None
        partial_mass = ad_partial_mass[index]
        barrier = ad_barrier[index]
        moments = ad_moments[index]
        diffusion = ad_diffusion[index]
        symmetry = ad_symmetry[index]

        if dof_type == 'rot':
            # 'Rot' specifies a mode corresponding to a hindered rotator. This requires additional inputs in the item file
            # the height of the barrier is fed through item.barrier_height
            # the internal inertial tensor cannot be calculated by the code and must be fed through item.custom_moments
            #     I have an auxillary script that people can use for this, but it involves changing constraints in a non-automated way per rotor
            moments_SI = moments * units._amu * 10**-20

            Q_rot = ( np.sqrt ( np.pi ) / symmetry) * ( ( 8.0*(np.pi)**2*moments_SI* units._k * temperature) /
                (units._hplanck**2))**(0.5)
            E_scale = barrier / ( 2 * units.kB * temperature )
            # calculation of the hindered barrier requires some imaginary numbers
            j_unit = 0+1j
            Q_hind = np.real( Q_rot * np.exp(-E_scale) * jn( 0 , j_unit*E_scale ) )
            S_tot += units.kB * ( np.log(Q_hind) )

        elif dof_type == 'rigid':
            # advanced_entropy is going to eventually replace the current 'RB' tag since it is more general
            # in order to calculate on a per-degree of freedom basis additional inputs are required
            # first the custom principle rotor can now be specified by the user for better consistency
            #   this comes from item.barrier_direction
            # no custom moments are needed since the full tensor is calculated for a rigid body
            if direction == 'x':
                decomp_intert = inertias[0]
            if direction == 'y':
                decomp_intert = inertias[1]
            if direction == 'z':
                decomp_intert = inertias[2]
            else:
                raise Exception("bad things")
            S_tmp = np.sqrt(np.pi * decomp_intert ) / item.sigma
            S_tmp *= (8.0 * np.pi**2 * units._k * temperature /
                units._hplanck**2)**(1.0 / 2.0)
            S_tot += units.kB * (np.log(S_tmp))

        elif dof_type == 'trans':
            # This comes directly from Tej's new paper, I am not as familiar and will talk to him
            # This assumes a 2-D translator, thus there is a Trans and Trans2 tag which specify the 2 modes to replace with effectively one mode of this method
            # This uses item.barrier_height as well as some imaginary numbers

            L = diffusion * 1E-10
            kT = units._k * temperature
            red_mass = partial_mass * units._amu #SI units
            T = temperature

            trans_bar= barrier * units._e #assuming barrier is in eV

            trans_freq=(trans_bar/(2* red_mass * L**2))**0.5
            #print(trans_freq/(3*10**10))

            r_x = trans_bar / (units._hplanck * trans_freq)
            T_x = units._k * T / (units._hplanck * trans_freq)
            r_T = r_x / T_x

            q_num = np.pi * r_T * np.exp(-r_T) * np.exp(-1/T_x) * (i0(r_T/2))**2 * np.exp(2.0/((2.0+16.0*r_x)*T_x))
            q_den = (1-np.exp(-1/T_x))**2

            qtrans = (q_num / q_den)**0.5
            #S_tmp = (2*np.pi*red_mass*kT*L**2/units._hplanck**2)**(1.0/2)
            S_tot += units.kB* ( np.log(qtrans) )

            # mass = sum(item.atoms.get_masses()) * units._amu  # kg/molecule
            # mass_Ar = 6.6335209E-26 # FIX LATER
            # mystery_a = 1E-10 # I don't know what this is, will talk to Tej
            # R = units.kB
            # j_unit = 0+1j
            # kT = units.kB * temperature
            # S_tmp = ( 2 / 3 ) * ( 18.6 * R + R * np.log( mass / mass_Ar)**1.5 * (temperature / 298.15)**2.5)
            # E_scale = barrier / ( 2 * units.kB * temperature )
            # vx =  ( ( ( barrier / units.J ) ) / (2 * mass * mystery_a**2  ) )**(1./2)
            # E_x = units._hplanck * vx / kT
            # print barrier
            # print E_x
            # #print E_x
            # j_unit = 0+1j
            # S_tmp += np.real( np.log( np.exp( -2. * E_scale) * jn( 0 , j_unit*E_scale )**2) )
            # print S_tmp
            # S_tmp += np.real( E_scale * ( 1. - jn( 1 , j_unit*E_scale ) / jn( 0 , j_unit*E_scale ) ) +
            #     2 * np.log(E_x / np.exp(1.) ) +
            #     2 * ( np.exp(E_x) / (np.exp(E_x) - 1. ) - np.log(1. - np.exp(-E_x) ) ) )
            # print S_tmp

            # S_tot += S_tmp

        # The default, will calculate a harmonic oscillator entropy
        else:
            kT = units.kB * temperature
            x = energy / kT
            S_tot += units.kB * ( - np.log(1-np.exp(-x)) )

        # sum each degree of freedom and return the total entropy
        S = S_tot
    return S

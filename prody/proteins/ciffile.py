# -*- coding: utf-8 -*-
"""This module defines functions for parsing `mmCIF files`_.

.. _mmCIF files: http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html"""


from collections import OrderedDict
import os.path
import numpy as np

from prody.atomic import AtomGroup
from prody.atomic import flags
from prody.atomic import ATOMIC_FIELDS
from prody.utilities import openFile
from prody import LOGGER, SETTINGS
from prody.utilities.misctools import getMasses

from .localpdb import fetchPDB
from .starfile import parseSTARLines, StarDict, parseSTARSection
from .cifheader import getCIFHeaderDict
from .header import buildBiomolecules, assignSecstr, isHelix, isSheet

__all__ = ['parseMMCIFStream', 'parseMMCIF', 'writeMMCIFStream', 'writeMMCIF', 'writePDBXMMCIFStream', 'writePDBXMMCIF']


class MMCIFParseError(Exception):
    pass


_parseMMCIFdoc = """
    :arg title: title of the :class:`.AtomGroup` instance, default is the
        PDB filename or PDB identifier
    :type title: str

    :arg chain: chain identifiers for parsing specific chains, e.g.
        ``chain='A'``, ``chain='B'``, ``chain='DE'``, by default all
        chains are parsed
    :type chain: str

    :arg subset: a predefined keyword to parse subset of atoms, valid keywords
        are ``'calpha'`` (``'ca'``), ``'backbone'`` (``'bb'``), or **None**
        (read all atoms), e.g. ``subset='bb'``
    :type subset: str

    :arg model: model index or None (read all models), e.g. ``model=10``
    :type model: int, list

    :arg altloc: if a location indicator is passed, such as ``'A'`` or ``'B'``,
         only indicated alternate locations will be parsed as the single
         coordinate set of the AtomGroup, if *altloc* is set **True** all
         alternate locations will be parsed and each will be appended as a
         distinct coordinate set, default is ``"A"``
    :type altloc: str
    """

_writeMMCIFdoc = """

    :arg atoms: an object with atom and coordinate data

    :arg csets: coordinate set indices, default is all coordinate sets

    :arg beta: a list or array of number to be outputted in beta column

    :arg occupancy: a list or array of number to be outputted in occupancy
        column
 
    """

_PDBSubsets = {'ca': 'ca', 'calpha': 'ca', 'bb': 'bb', 'backbone': 'bb'}


def parseMMCIF(pdb, **kwargs):
    """Returns an :class:`.AtomGroup` and/or a :class:`.StarDict` containing header data
    parsed from an mmCIF file. If not found, the mmCIF file will be downloaded
    from the PDB. It will be downloaded in uncompressed format regardless of
    the compressed keyword.

    This function extends :func:`.parseMMCIFStream`.

    :arg pdb: a PDB identifier or a filename
        If needed, mmCIF files are downloaded using :func:`.fetchPDB()` function.
    :type pdb: str

    :arg chain: comma separated string or list-like of chain IDs
    :type chain: str, tuple, list, :class:`~numpy.ndarray`
    """
    chain = kwargs.pop('chain', None)
    title = kwargs.get('title', None)
    auto_bonds = SETTINGS.get('auto_bonds')
    get_bonds = kwargs.get('bonds', auto_bonds)
    if get_bonds:
        LOGGER.warn('Parsing struct_conn information from mmCIF is current unsupported and no bond information is added to the results')
    if not os.path.isfile(pdb):
        if len(pdb) == 5 and pdb.isalnum():
            if chain is None:
                chain = pdb[-1]
                pdb = pdb[:4]
            else:
                raise ValueError('Please provide chain as a keyword argument '
                                 'or part of the PDB ID, not both')
        else:
            chain = chain

        if len(pdb) == 4 and pdb.isalnum():
            if title is None:
                title = pdb
                kwargs['title'] = title

            if os.path.isfile(pdb + '.cif'):
                filename = pdb + '.cif'
            elif os.path.isfile(pdb + '.cif.gz'):
                filename = pdb + '.cif.gz'
            else:
                filename = fetchPDB(pdb, report=True,
                                    format='cif', compressed=False)
                if filename is None:
                    raise IOError('mmCIF file for {0} could not be downloaded.'
                                  .format(pdb))
            pdb = filename
        else:
            raise IOError('{0} is not a valid filename or a valid PDB '
                          'identifier.'.format(pdb))
    if title is None:
        title, ext = os.path.splitext(os.path.split(pdb)[1])
        if ext == '.gz':
            title, ext = os.path.splitext(title)
        if len(title) == 7 and title.startswith('pdb'):
            title = title[3:]
        kwargs['title'] = title
    cif = openFile(pdb, 'rt')
    result = parseMMCIFStream(cif, chain=chain, **kwargs)
    cif.close()
    return result


def parseMMCIFStream(stream, **kwargs):
    """Returns an :class:`.AtomGroup` and/or a class:`.StarDict` 
    containing header data parsed from a stream of CIF lines.

    :arg stream: Anything that implements the method ``readlines``
        (e.g. :class:`file`, buffer, stdin)
        
    """

    model = kwargs.get('model')
    subset = kwargs.get('subset')
    chain = kwargs.get('chain')
    altloc = kwargs.get('altloc', 'A')
    header = kwargs.get('header', False)
    parsing_class = kwargs.get('parsing_class', 'auth')
    assert isinstance(header, bool), 'header must be a boolean'

    if model is not None:
        if isinstance(model, int):
            if model < 0:
                raise ValueError('model must be greater than 0')
        else:
            raise TypeError('model must be an integer, {0} is invalid'
                            .format(str(model)))
    title_suffix = ''
    if subset:
        try:
            subset = _PDBSubsets[subset.lower()]
        except AttributeError:
            raise TypeError('subset must be a string')
        except KeyError:
            raise ValueError('{0} is not a valid subset'
                             .format(repr(subset)))
        title_suffix = '_' + subset
    if chain is not None:
        if not isinstance(chain, str):
            raise TypeError('chain must be a string')
        elif len(chain) == 0:
            raise ValueError('chain must not be an empty string')
        title_suffix = '_' + chain + title_suffix

    ag = None
    if 'ag' in kwargs:
        ag = kwargs['ag']
        if not isinstance(ag, AtomGroup):
            raise TypeError('ag must be an AtomGroup instance')
        n_csets = ag.numCoordsets()
    elif model != 0:
        ag = AtomGroup(str(kwargs.get('title', 'Unknown')) + title_suffix)
        n_csets = 0

    biomol = kwargs.get('biomol', False)
    auto_secondary = SETTINGS.get('auto_secondary')
    secondary = kwargs.get('secondary', auto_secondary)
    hd = None
    if model != 0:
        LOGGER.timeit()
        try:
            lines = stream.readlines()
        except AttributeError as err:
            try:
                lines = stream.read().split('\n')
            except AttributeError:
                raise err
        if not len(lines):
            raise ValueError('empty PDB file or stream')
        if header or biomol or secondary:
            hd = getCIFHeaderDict(lines)

        _parseMMCIFLines(ag, lines, model, chain, subset, altloc, parsing_class)

        if ag.numAtoms() > 0:
            LOGGER.report('{0} atoms and {1} coordinate set(s) were '
                          'parsed in %.2fs.'.format(ag.numAtoms(),
                                                    ag.numCoordsets() - n_csets))
        else:
            ag = None
            LOGGER.warn('Atomic data could not be parsed, please '
                        'check the input file.')
    elif header:
        hd = getCIFHeaderDict(stream)

    if ag is not None and isinstance(hd, dict):
        if secondary:
            if auto_secondary:
                try:
                    ag = assignSecstr(hd, ag)
                except ValueError:
                    pass
            else:
                ag = assignSecstr(hd, ag)
        if biomol:
            ag = buildBiomolecules(hd, ag)

            if isinstance(ag, list):
                LOGGER.info('Biomolecular transformations were applied, {0} '
                            'biomolecule(s) are returned.'.format(len(ag)))
            else:
                LOGGER.info('Biomolecular transformations were applied to the '
                            'coordinate data.')    

    if model != 0:
        if header:
            return ag, hd
        else:
            return ag
    else:
        return hd


parseMMCIFStream.__doc__ += _parseMMCIFdoc


def _parseMMCIFLines(atomgroup, lines, model, chain, subset,
                     altloc_torf, parsing_class):
    """Returns an AtomGroup. See also :func:`.parsePDBStream()`.

    :arg lines: mmCIF lines
    """

    if subset is not None:
        if subset == 'ca':
            subset = set(('CA',))
        elif subset in 'bb':
            subset = flags.BACKBONE
        protein_resnames = flags.AMINOACIDS

    asize = 0
    i = 0
    models = []
    nModels = 0
    fields = OrderedDict()
    fieldCounter = -1
    foundAtomBlock = False
    doneAtomBlock = False
    start = 0
    stop = 0
    while not doneAtomBlock:
        line = lines[i]
        if line[:11] == '_atom_site.':
            fieldCounter += 1
            fields[line.split('.')[1].strip()] = fieldCounter

        if line.startswith('ATOM ') or line.startswith('HETATM'):
            if not foundAtomBlock:
                foundAtomBlock = True
                start = i
            models.append(line.split()[fields['pdbx_PDB_model_num']])
            if len(models) == 1 or (models[asize] != models[asize-1]):
                nModels += 1
            asize += 1
        else:
            if foundAtomBlock:
                doneAtomBlock = True
                stop = i
        i += 1

    new_start = start
    new_stop = stop
    if model is not None:
        startp1 = start + 1
        for i in range(start, stop):
            if int(models[i-startp1]) != model and int(models[i+1-startp1]) == model:
                new_start = i
            if model != nModels and (int(models[i-startp1]) == model and int(models[i+1-startp1]) != model):
                new_stop = i
                break
        if not str(model) in models:
            raise MMCIFParseError('model {0} is not found'.format(model))

    start = new_start
    stop = new_stop

    addcoords = False
    if atomgroup.numCoordsets() > 0:
        addcoords = True

    if isinstance(altloc_torf, str):
        if altloc_torf == 'all':
            which_altlocs = 'all'
        elif altloc_torf == 'first':
            which_altlocs = 'first'
        elif altloc_torf.strip() != 'A':
            LOGGER.info('Parsing alternate locations {0}.'
                        .format(altloc_torf))
            which_altlocs = '.' + ''.join(altloc_torf.split())
        else:
            which_altlocs = '.A'
        altloc_torf = False
    else:
        which_altlocs = '.A'
        altloc_torf = True

    coordinates = np.zeros((asize, 3), dtype=float)
    auth_atomnames = np.zeros(asize, dtype=ATOMIC_FIELDS['name'].dtype)
    label_atomnames = np.zeros(asize, dtype=ATOMIC_FIELDS['label_name'].dtype)
    auth_resnames = np.zeros(asize, dtype=ATOMIC_FIELDS['resname'].dtype)
    label_resnames = np.zeros(asize, dtype=ATOMIC_FIELDS['label_resname'].dtype)
    auth_resnums = np.zeros(asize, dtype=ATOMIC_FIELDS['resnum'].dtype)
    label_resnums = np.zeros(asize, dtype=ATOMIC_FIELDS['label_resnum'].dtype)
    auth_chainids = np.zeros(asize, dtype=ATOMIC_FIELDS['chain'].dtype)
    label_chainids = np.zeros(asize, dtype=ATOMIC_FIELDS['label_chain'].dtype)
    segnames = np.zeros(asize, dtype=ATOMIC_FIELDS['segment'].dtype)
    hetero = np.zeros(asize, dtype=bool)
    termini = np.zeros(asize, dtype=bool)
    altlocs = np.zeros(asize, dtype=ATOMIC_FIELDS['altloc'].dtype)
    icodes = np.zeros(asize, dtype=ATOMIC_FIELDS['icode'].dtype)
    serials = np.zeros(asize, dtype=ATOMIC_FIELDS['serial'].dtype)
    elements = np.zeros(asize, dtype=ATOMIC_FIELDS['element'].dtype)
    bfactors = np.zeros(asize, dtype=ATOMIC_FIELDS['beta'].dtype)
    occupancies = np.zeros(asize, dtype=ATOMIC_FIELDS['occupancy'].dtype)
    charges = np.zeros(asize, dtype=ATOMIC_FIELDS['charge'].dtype)

    entity_ids = np.zeros(asize, dtype=ATOMIC_FIELDS['entity_id'].dtype)

    n_atoms = atomgroup.numAtoms()
    if n_atoms > 0:
        asize = n_atoms

    acount = 0

    resnum_tmp = ""
    resname_tmp = ""
    list_atomnames_by_res = []

    for i, line in enumerate(lines[start:stop]):
        startswith = line.split()[fields['group_PDB']]

        #Atomname
        auth_atomname = line.split()[fields['auth_atom_id']]
        #if auth_atomname.startswith('"') and auth_atomname.endswith('"'):
        #    auth_atomname = auth_atomname[1:-1]
        label_atomname = line.split()[fields['label_atom_id']]
        #if label_atomname.startswith('"') and label_atomname.endswith('"'):
        #    label_atomname = label_atomname[1:-1]

        if parsing_class == "auth":
            atomname = auth_atomname
        else:
            atomname = label_atomname

        #Resname
        auth_resname = line.split()[fields['auth_comp_id']]
        label_resname = line.split()[fields['label_comp_id']]
        if parsing_class == "auth":
            resname = auth_resname
        else:
            resname = label_resname

        #Select subset
        if subset is not None:
            if not (atomname in subset and resname in protein_resnames):
                continue

        #chID
        auth_chID = line.split()[fields['auth_asym_id']]
        label_chID = line.split()[fields['label_asym_id']]
        if parsing_class == "auth":
            chID = auth_chID
        else:
            chID = label_chID
        
        if chain is not None:
            if isinstance(chain, str):
                chain = chain.split(',')
            if not chID in chain:
                continue

        #Resnum
        auth_resnum = line.split()[fields['auth_seq_id']]
        label_resnum = line.split()[fields['label_seq_id']]

        #Altloc
        alt = line.split()[fields['label_alt_id']]
        if alt not in which_altlocs and which_altlocs != 'all' and which_altlocs != 'first':
            continue
        elif which_altlocs == 'first':    #Select the 1st alternate location (PDB:5DXX-res268)
            #Same residue
            if label_resnum == resnum_tmp and resname == resname_tmp:
                if atomname not in list_atomnames_by_res:
                    list_atomnames_by_res.append(atomname)
                else:
                    continue
            #Two consecutive residues
            elif label_resnum != resnum_tmp:
                resnum_tmp = label_resnum
                resname_tmp = resname
                list_atomnames_by_res = [atomname]
            #Alternate location problem (resnum equal and different resname)
            else:
                continue

        if label_resnum == '.':
            label_resnum = "0"

        coordinates[acount] = [line.split()[fields['Cartn_x']],
                               line.split()[fields['Cartn_y']],
                               line.split()[fields['Cartn_z']]]
        auth_atomnames[acount] = auth_atomname
        label_atomnames[acount] = label_atomname
        auth_resnames[acount] = auth_resname
        label_resnames[acount] = label_resname
        auth_resnums[acount] = auth_resnum
        label_resnums[acount] = label_resnum
        auth_chainids[acount] = auth_chID
        label_chainids[acount] = label_chID
        segnames[acount] = label_chID
        hetero[acount] = startswith == 'HETATM' # True or False

        if parsing_class == "auth":
            if auth_chainids[acount] != auth_chainids[acount-1]: 
                termini[acount-1] = True
        else:
            if label_chainids[acount] != label_chainids[acount-1]: 
                termini[acount-1] = True

        if alt == '.':
            alt = ' '
        altlocs[acount] = alt

        try:
            icodes[acount] = line.split()[fields['pdbx_PDB_ins_code']]
        except KeyError:
            icodes[acount] = ''

        if icodes[acount] == '?': 
            icodes[acount] = ''

        serials[acount] = line.split()[fields['id']]
        elements[acount] = line.split()[fields['type_symbol']]
        bfactors[acount] = line.split()[fields['B_iso_or_equiv']]
        occupancies[acount] = line.split()[fields['occupancy']]

        #Get charge
        charge = line.split()[fields['pdbx_formal_charge']]
        if charge == '?':
            charges[acount] = 0
        else:
            try:
                charges[acount] = int(charge)
            except:
                try:
                    charges[acount] = int(charge[1] + charge[0])
                except:
                    charges[acount] = 0
                    LOGGER.warn('failed to parse charge at line {0}'
                                .format(i))

        entity_ids[acount] = line.split()[fields['label_entity_id']]

        acount += 1

    if model is None:
        modelSize = acount//nModels
    else:
        modelSize = acount

    if addcoords:
        atomgroup.addCoordset(coordinates[:modelSize])
    else:
        atomgroup._setCoords(coordinates[:modelSize])

    atomgroup.setNames(auth_atomnames[:modelSize])
    atomgroup.setLabelNames(label_atomnames[:modelSize])
    atomgroup.setResnames(auth_resnames[:modelSize])
    atomgroup.setLabelResnames(label_resnames[:modelSize])
    atomgroup.setResnums(auth_resnums[:modelSize])
    atomgroup.setLabelResnums(label_resnums[:modelSize])
    atomgroup.setSegnames(segnames[:modelSize])
    atomgroup.setChids(auth_chainids[:modelSize])
    atomgroup.setLabelChids(label_chainids[:modelSize])

    atomgroup.setFlags('hetatm', hetero[:modelSize])
    atomgroup.setFlags('pdbter', termini[:modelSize])
    atomgroup.setAltlocs(altlocs[:modelSize])
    atomgroup.setIcodes(icodes[:modelSize])
    atomgroup.setSerials(serials[:modelSize])

    atomgroup.setElements(elements[:modelSize])
    atomgroup.setMasses(getMasses(elements[:modelSize]))
    atomgroup.setBetas(bfactors[:modelSize])
    atomgroup.setOccupancies(occupancies[:modelSize])

    atomgroup.setCharges(charges[:modelSize])
    atomgroup.setEntityID(entity_ids[:modelSize])

    anisou = None
    siguij = None
    try:
        data = parseSTARSection(lines, "_atom_site_anisotrop")
        x = data[0] # check if data has anything in it
    except IndexError:
        LOGGER.warn("No anisotropic B factors found")
    else:
        anisou = np.zeros((acount, 6),
                        dtype=ATOMIC_FIELDS['anisou'].dtype)

        if "_atom_site_anisotrop.U[1][1]_esd" in data[0].keys():
            siguij = np.zeros((acount, 6),
                            dtype=ATOMIC_FIELDS['siguij'].dtype)

        for entry in data:
            try:
                index = np.where(atomgroup.getSerials() == int(
                    entry["_atom_site_anisotrop.id"]))[0][0]
            except:
                continue
                
            anisou[index, 0] = entry['_atom_site_anisotrop.U[1][1]']
            anisou[index, 1] = entry['_atom_site_anisotrop.U[2][2]']
            anisou[index, 2] = entry['_atom_site_anisotrop.U[3][3]']
            anisou[index, 3] = entry['_atom_site_anisotrop.U[1][2]']
            anisou[index, 4] = entry['_atom_site_anisotrop.U[1][3]']
            anisou[index, 5] = entry['_atom_site_anisotrop.U[2][3]'] 

            if siguij is not None:
                try:
                    siguij[index, 0] = entry['_atom_site_anisotrop.U[1][1]_esd']
                    siguij[index, 1] = entry['_atom_site_anisotrop.U[2][2]_esd']
                    siguij[index, 2] = entry['_atom_site_anisotrop.U[3][3]_esd']
                    siguij[index, 3] = entry['_atom_site_anisotrop.U[1][2]_esd']
                    siguij[index, 4] = entry['_atom_site_anisotrop.U[1][3]_esd']
                    siguij[index, 5] = entry['_atom_site_anisotrop.U[2][3]_esd']
                except:
                    pass

        atomgroup.setAnisous(anisou) # no division needed anymore

        if not np.any(siguij):
            atomgroup.setAnistds(siguij)  # no division needed anymore

    if model is None:
        for n in range(1, nModels):
            atomgroup.addCoordset(coordinates[n*modelSize:(n+1)*modelSize])

    return atomgroup


def writeMMCIF(filename, atoms, csets=None, **kwargs):
    """Write *atoms* in MMCIF format to a file with name *filename* and return
    *filename*.  If *filename* ends with :file:`.gz`, a compressed file will
    be written.

    :arg renumber: whether to renumber atoms with serial indices
        Default is **True**
    :type renumber: bool
    """
    if not ('.cif' in filename or '.cif.gz' in filename or '.mmcif' in filename or '.mmcif.gz' in filename):
        filename += '.cif'
    out = openFile(filename, 'wt')
    writeMMCIFStream(out, atoms, csets, **kwargs)
    out.close()
    return filename


def writeMMCIFStream(stream, atoms, csets=None, **kwargs):
    """Write *atoms* in MMCIF format to a *stream*.

    :arg stream: anything that implements a :meth:`write` method (e.g. file,
        buffer, stdout)

    :arg renumber: whether to renumber atoms with serial indices
        Default is **True**
    :type renumber: bool
    """
    renumber = kwargs.get('renumber', True)
    anisous_flag = kwargs.get('anisous_flag', False)

    remark = str(atoms)
    try:
        coordsets = atoms.getCoordsets(csets)
    except AttributeError:
        try:
            coordsets = atoms.getCoords()
        except AttributeError:
            raise TypeError('atoms must be an object with coordinate sets')
        if coordsets is not None:
            coordsets = [coordsets]
    else:
        if coordsets.ndim == 2:
            coordsets = [coordsets]
    if coordsets is None:
        raise ValueError('atoms does not have any coordinate sets')

    try:
        acsi = atoms.getACSIndex()
    except AttributeError:
        try:
            atoms = atoms.getAtoms()
        except AttributeError:
            raise TypeError('atoms must be an Atomic instance or an object '
                            'with `getAtoms` method')
        else:
            if atoms is None:
                raise ValueError('atoms is not associated with an Atomic '
                                 'instance')
            try:
                acsi = atoms.getACSIndex()
            except AttributeError:
                raise TypeError('atoms does not have a valid type')

    try:
        atoms.getIndex()
    except AttributeError:
        pass
    else:
        atoms = atoms.select('all')

    n_atoms = atoms.numAtoms()

    occupancy = kwargs.get('occupancy')
    if occupancy is None:
        occupancies = atoms.getOccupancies()
        if occupancies is None:
            occupancies = np.zeros(n_atoms, float)
    else:
        occupancies = np.array(occupancy)
        if len(occupancies) != n_atoms:
            raise ValueError('len(occupancy) must be equal to number of atoms')

    beta = kwargs.get('beta')
    if beta is None:
        bfactors = atoms.getBetas()
        if bfactors is None:
            bfactors = np.zeros(n_atoms, float)
    else:
        bfactors = np.array(beta)
        if len(bfactors) != n_atoms:
            raise ValueError('len(beta) must be equal to number of atoms')

    atomnames = atoms.getNames()
    if atomnames is None:
        raise ValueError('atom names are not set')

    label_atomnames = atoms.getLabelNames()
    if label_atomnames is None:
        label_atomnames = ['X'] * n_atoms

    s_or_u = np.array(['a']).dtype.char

    altlocs = atoms.getAltlocs()
    if altlocs is None:
        altlocs = np.zeros(n_atoms, s_or_u + '1')

    resnames = atoms.getResnames()
    if resnames is None:
        resnames = ['UNK'] * n_atoms

    label_resnames = atoms.getLabelResnames()
    if label_resnames is None:
        label_resnames = ['UNK'] * n_atoms

    chainids = atoms.getChids()
    if chainids is None:
        chainids = np.zeros(n_atoms, s_or_u + '1')

    label_chainids = atoms.getLabelChids()
    if label_chainids is None:
        label_chainids = np.zeros(n_atoms, s_or_u + '1')

    resnums = atoms.getResnums()
    if resnums is None:
        resnums = np.ones(n_atoms, int)

    label_resnums = atoms.getLabelResnums()
    if label_resnums is None:
        label_resnums = ["1"] * n_atoms
    else:
        label_resnums = np.array(label_resnums).astype('str')

    serials = atoms.getSerials()
    if serials is None or renumber:
        serials = np.arange(n_atoms, dtype=int) + 1

    icodes = atoms.getIcodes()
    if icodes is None:
        icodes = np.zeros(n_atoms, s_or_u + '1')

    hetero = ['ATOM'] * n_atoms
    heteroflags = atoms.getFlags('hetatm')
    if heteroflags is None:
        heteroflags = atoms.getFlags('hetero')
    if heteroflags is not None:
        hetero = np.array(hetero, s_or_u + '6')
        hetero[heteroflags] = 'HETATM'

    elements = atoms.getElements()
    if elements is None:
        elements = np.zeros(n_atoms, s_or_u + '1')

    charges = atoms.getCharges()
    charges2 = np.empty(n_atoms, s_or_u + '2')
    if charges is not None:
        for i, charge in enumerate(charges):
            charges2[i] = str(abs(int(charge)))

            if np.sign(charge) == -1:
                charges2[i] = '-' + charges2[i]
            else:
                charges2[i] = '+' + charges2[i]

            if charges2[i] == '+0':
                charges2[i] = '?'

    anisous = atoms.getAnisous()

    entity_ids = atoms.getEntityID()
    if entity_ids is None:
        entity_ids = [1] * n_atoms

    # write file
    write = stream.write
    
    # write remarks
    write('# {0}\n'.format(remark))
    write("#\n")
    write(f"data_{remark.split()[-1]}\n")
    write(f"_entry.id {remark.split()[-1]}\n")
    write("#\n")

    # write atoms
    #Number of NMR sets
    nb_sets = len(coordsets)
    nb_atoms = len(coordsets[0])

    list_group_PDB = list(hetero) * nb_sets
    list_id = list(serials) * nb_sets
    list_type_symbol = list(elements) * nb_sets
    list_label_atom_id = list(label_atomnames) * nb_sets
    altlocs[altlocs == " "] = "."
    list_label_alt_id = list(altlocs) * nb_sets
    list_label_comp_id = list(label_resnames) * nb_sets
    label_chainids[label_chainids == ""] = "."
    list_label_asym_id = list(label_chainids) * nb_sets
    list_label_entity_id = list(entity_ids) * nb_sets
    label_resnums[label_resnums == "0"] = "."
    list_label_seq_id = list(label_resnums) * nb_sets
    icodes[icodes == ""] = "?"
    list_pdbx_PDB_ins_code = list(icodes) * nb_sets
    list_occupancy = list(occupancies) * nb_sets
    list_B_iso_or_equiv = list(bfactors) * nb_sets
    list_pdbx_formal_charge = list(charges2) * nb_sets
    list_auth_asym_id = list(chainids) * nb_sets

    list_Cartn_x = []
    list_Cartn_y = []
    list_Cartn_z = []
    list_pdbx_PDB_model_num = []
    for m, coords in enumerate(coordsets):
        list_Cartn_x += list(coords[:,0])
        list_Cartn_y += list(coords[:,1])
        list_Cartn_z += list(coords[:,2])
        list_pdbx_PDB_model_num += [m+1] * nb_atoms


    #Get the column widths
    max_group_PDB = len(max(list_group_PDB, key=len))
    max_id = len(str(max(list_id)))
    max_type_symbol = len(max(list_type_symbol, key=len))
    max_label_atom_id = len(max(list_label_atom_id, key=len))
    max_label_alt_id = len(max(list_label_alt_id, key=len))
    max_label_comp_id = len(max(list_label_comp_id, key=len))
    max_label_asym_id = len(max(list_label_asym_id, key=len))
    max_label_entity_id = len(str(max(list_label_entity_id)))
    max_label_seq_id = len(max(list_label_seq_id, key=len))
    max_pdbx_PDB_ins_code = len(max(list_pdbx_PDB_ins_code, key=len))
    max_Cartn_x = max([len(str(x).split(".")[0]) for x in list_Cartn_x]) + 4
    max_Cartn_y = max([len(str(x).split(".")[0]) for x in list_Cartn_y]) + 4
    max_Cartn_z = max([len(str(x).split(".")[0]) for x in list_Cartn_z]) + 4
    max_occupancy = 4
    max_B_iso_or_equiv = len(str(max(list_B_iso_or_equiv)).split(".")[0]) + 3
    max_pdbx_formal_charge = len(max(list_pdbx_formal_charge, key=len))
    max_auth_asym_id = len(max(list_auth_asym_id, key=len))
    max_pdbx_PDB_model_num = len(str(max(list_pdbx_PDB_model_num)))

    write("loop_\n"
          "_atom_site.group_PDB \n"
          "_atom_site.id \n"
          "_atom_site.type_symbol \n"
          "_atom_site.label_atom_id \n"
          "_atom_site.label_alt_id \n"
          "_atom_site.label_comp_id \n"
          "_atom_site.label_asym_id \n"
          "_atom_site.label_entity_id \n"
          "_atom_site.label_seq_id \n"
          "_atom_site.pdbx_PDB_ins_code \n"
          "_atom_site.Cartn_x \n"
          "_atom_site.Cartn_y \n"
          "_atom_site.Cartn_z \n"
          "_atom_site.occupancy \n"
          "_atom_site.B_iso_or_equiv \n"
          "_atom_site.pdbx_formal_charge \n"
          "_atom_site.auth_asym_id \n"
          "_atom_site.pdbx_PDB_model_num \n")

    for i, _ in enumerate(list_group_PDB):
        write(f"{list_group_PDB[i]:<{max_group_PDB}s} "
              f"{list_id[i]:<{max_id}d} "
              f"{list_type_symbol[i]:<{max_type_symbol}s} "
              f"{list_label_atom_id[i]:<{max_label_atom_id}s} "
              f"{list_label_alt_id[i]:<{max_label_alt_id}s} "
              f"{list_label_comp_id[i]:<{max_label_comp_id}s} "
              f"{list_label_asym_id[i]:<{max_label_asym_id}s} "
              f"{list_label_entity_id[i]:<{max_label_entity_id}d} "
              f"{list_label_seq_id[i]:<{max_label_seq_id}s} "
              f"{list_pdbx_PDB_ins_code[i]:<{max_pdbx_PDB_ins_code}s} "
              f"{list_Cartn_x[i]:<{max_Cartn_x}.3f} "
              f"{list_Cartn_y[i]:<{max_Cartn_y}.3f} "
              f"{list_Cartn_z[i]:<{max_Cartn_z}.3f} "
              f"{list_occupancy[i]:<{max_occupancy}.2f} "
              f"{list_B_iso_or_equiv[i]:<{max_B_iso_or_equiv}.2f} "
              f"{list_pdbx_formal_charge[i]:<{max_pdbx_formal_charge}s} "
              f"{list_auth_asym_id[i]:<{max_auth_asym_id}s} "
              f"{list_pdbx_PDB_model_num[i]:<{max_pdbx_PDB_model_num}d} \n")
    write("#")

    if anisous is not None and anisous_flag:
        anisotrop_U11 = anisous[:,0]
        anisotrop_U22 = anisous[:,1]
        anisotrop_U33 = anisous[:,2]
        anisotrop_U12 = anisous[:,3]
        anisotrop_U13 = anisous[:,4]
        anisotrop_U23 = anisous[:,5]
        max_anisotrop_U11 = max([len(str(x).split(".")[0]) for x in anisotrop_U11]) + 5
        max_anisotrop_U22 = max([len(str(x).split(".")[0]) for x in anisotrop_U22]) + 5
        max_anisotrop_U33 = max([len(str(x).split(".")[0]) for x in anisotrop_U33]) + 5
        max_anisotrop_U12 = max([len(str(x).split(".")[0]) for x in anisotrop_U12]) + 5
        max_anisotrop_U13 = max([len(str(x).split(".")[0]) for x in anisotrop_U13]) + 5
        max_anisotrop_U23 = max([len(str(x).split(".")[0]) for x in anisotrop_U23]) + 5

        write("loop_\n"
              "_atom_site_anisotrop.id \n"
              "_atom_site_anisotrop.type_symbol \n"
              "_atom_site_anisotrop.pdbx_label_atom_id \n"
              "_atom_site_anisotrop.pdbx_label_alt_id \n"
              "_atom_site_anisotrop.pdbx_label_comp_id \n"
              "_atom_site_anisotrop.pdbx_label_asym_id \n"
              "_atom_site_anisotrop.pdbx_label_seq_id \n"
              "_atom_site_anisotrop.pdbx_PDB_ins_code \n"
              "_atom_site_anisotrop.U[1][1] \n"
              "_atom_site_anisotrop.U[2][2] \n"
              "_atom_site_anisotrop.U[3][3] \n"
              "_atom_site_anisotrop.U[1][2] \n"
              "_atom_site_anisotrop.U[1][3] \n"
              "_atom_site_anisotrop.U[2][3] \n"
              "_atom_site_anisotrop.pdbx_auth_asym_id \n")

        for i, _ in enumerate(list_group_PDB):
            if anisotrop_U11[i] != 0 and anisotrop_U22[i] != 0 and anisotrop_U33[i] != 0 and anisotrop_U12[i] != 0 and anisotrop_U13[i] != 0 and anisotrop_U23[i] != 0:
                write(f"{list_id[i]:<{max_id}d} "
                    f"{list_type_symbol[i]:<{max_type_symbol}s} "
                    f"{list_label_atom_id[i]:<{max_label_atom_id}s} "
                    f"{list_label_alt_id[i]:<{max_label_alt_id}s} "
                    f"{list_label_comp_id[i]:<{max_label_comp_id}s} "
                    f"{list_label_asym_id[i]:<{max_label_asym_id}s} "
                    f"{list_label_seq_id[i]:<{max_label_seq_id}s} "
                    f"{list_pdbx_PDB_ins_code[i]:<{max_pdbx_PDB_ins_code}s} "
                    f"{anisotrop_U11[i]:<{max_anisotrop_U11}.4f} "
                    f"{anisotrop_U22[i]:<{max_anisotrop_U22}.4f} "
                    f"{anisotrop_U33[i]:<{max_anisotrop_U33}.4f} "
                    f"{anisotrop_U12[i]:<{max_anisotrop_U12}.4f} "
                    f"{anisotrop_U13[i]:<{max_anisotrop_U13}.4f} "
                    f"{anisotrop_U23[i]:<{max_anisotrop_U23}.4f} "
                    f"{list_auth_asym_id[i]:<{max_auth_asym_id}s} \n")
        write("#")


def writePDBXMMCIF(filename, atoms, csets=None, **kwargs):
    """Write *atoms* in PDBXMMCIF format to a file with name *filename* and return
    *filename*.  If *filename* ends with :file:`.gz`, a compressed file will
    be written.

    :arg renumber: whether to renumber atoms with serial indices
        Default is **True**
    :type renumber: bool
    """
    if not ('.cif' in filename or '.cif.gz' in filename or '.mmcif' in filename or '.mmcif.gz' in filename):
        filename += '.cif'
    out = openFile(filename, 'wt')
    writePDBXMMCIFStream(out, atoms, csets, **kwargs)
    out.close()
    return filename


def writePDBXMMCIFStream(stream, atoms, csets=None, **kwargs):
    """Write *atoms* in PDBXMMCIF format to a *stream*.

    :arg stream: anything that implements a :meth:`write` method (e.g. file,
        buffer, stdout)

    :arg renumber: whether to renumber atoms with serial indices
        Default is **True**
    :type renumber: bool
    """
    renumber = kwargs.get('renumber', True)
    anisous_flag = kwargs.get('anisous_flag', False)

    remark = str(atoms)
    try:
        coordsets = atoms.getCoordsets(csets)
    except AttributeError:
        try:
            coordsets = atoms.getCoords()
        except AttributeError:
            raise TypeError('atoms must be an object with coordinate sets')
        if coordsets is not None:
            coordsets = [coordsets]
    else:
        if coordsets.ndim == 2:
            coordsets = [coordsets]
    if coordsets is None:
        raise ValueError('atoms does not have any coordinate sets')

    try:
        acsi = atoms.getACSIndex()
    except AttributeError:
        try:
            atoms = atoms.getAtoms()
        except AttributeError:
            raise TypeError('atoms must be an Atomic instance or an object '
                            'with `getAtoms` method')
        else:
            if atoms is None:
                raise ValueError('atoms is not associated with an Atomic '
                                 'instance')
            try:
                acsi = atoms.getACSIndex()
            except AttributeError:
                raise TypeError('atoms does not have a valid type')

    try:
        atoms.getIndex()
    except AttributeError:
        pass
    else:
        atoms = atoms.select('all')

    n_atoms = atoms.numAtoms()

    occupancy = kwargs.get('occupancy')
    if occupancy is None:
        occupancies = atoms.getOccupancies()
        if occupancies is None:
            occupancies = np.zeros(n_atoms, float)
    else:
        occupancies = np.array(occupancy)
        if len(occupancies) != n_atoms:
            raise ValueError('len(occupancy) must be equal to number of atoms')

    beta = kwargs.get('beta')
    if beta is None:
        bfactors = atoms.getBetas()
        if bfactors is None:
            bfactors = np.zeros(n_atoms, float)
    else:
        bfactors = np.array(beta)
        if len(bfactors) != n_atoms:
            raise ValueError('len(beta) must be equal to number of atoms')

    atomnames = atoms.getNames()
    if atomnames is None:
        raise ValueError('atom names are not set')

    label_atomnames = atoms.getLabelNames()
    if label_atomnames is None:
        label_atomnames = ['X'] * n_atoms

    s_or_u = np.array(['a']).dtype.char

    altlocs = atoms.getAltlocs()
    if altlocs is None:
        altlocs = np.zeros(n_atoms, s_or_u + '1')

    resnames = atoms.getResnames()
    if resnames is None:
        resnames = ['UNK'] * n_atoms

    label_resnames = atoms.getLabelResnames()
    if label_resnames is None:
        label_resnames = ['UNK'] * n_atoms

    chainids = atoms.getChids()
    if chainids is None:
        chainids = np.zeros(n_atoms, s_or_u + '1')

    label_chainids = atoms.getLabelChids()
    if label_chainids is None:
        label_chainids = np.zeros(n_atoms, s_or_u + '1')

    resnums = atoms.getResnums()
    if resnums is None:
        resnums = np.ones(n_atoms, int)

    label_resnums = atoms.getLabelResnums()
    if label_resnums is None:
        label_resnums = ["1"] * n_atoms
    else:
        label_resnums = np.array(label_resnums).astype('str')

    serials = atoms.getSerials()
    if serials is None or renumber:
        serials = np.arange(n_atoms, dtype=int) + 1

    icodes = atoms.getIcodes()
    if icodes is None:
        icodes = np.zeros(n_atoms, s_or_u + '1')

    hetero = ['ATOM'] * n_atoms
    heteroflags = atoms.getFlags('hetatm')
    if heteroflags is None:
        heteroflags = atoms.getFlags('hetero')
    if heteroflags is not None:
        hetero = np.array(hetero, s_or_u + '6')
        hetero[heteroflags] = 'HETATM'

    elements = atoms.getElements()
    if elements is None:
        elements = np.zeros(n_atoms, s_or_u + '1')

    charges = atoms.getCharges()
    charges2 = np.empty(n_atoms, s_or_u + '2')
    if charges is not None:
        for i, charge in enumerate(charges):
            charges2[i] = str(abs(int(charge)))

            if np.sign(charge) == -1:
                charges2[i] = '-' + charges2[i]
            else:
                charges2[i] = '+' + charges2[i]

            if charges2[i] == '+0':
                charges2[i] = '?'

    anisous = atoms.getAnisous()

    entity_ids = atoms.getEntityID()
    if entity_ids is None:
        entity_ids = [1] * n_atoms

    # write file
    write = stream.write
    
    # write remarks
    write('# {0}\n'.format(remark))
    write("#\n")
    write(f"data_{remark.split()[-1]}\n")
    write(f"_entry.id {remark.split()[-1]}\n")
    write("#\n")

    # write atoms
    #Number of NMR sets
    nb_sets = len(coordsets)
    nb_atoms = len(coordsets[0])

    list_group_PDB = list(hetero) * nb_sets
    list_id = list(serials) * nb_sets
    list_type_symbol = list(elements) * nb_sets
    list_label_atom_id = list(label_atomnames) * nb_sets
    altlocs[altlocs == " "] = "."
    list_label_alt_id = list(altlocs) * nb_sets
    list_label_comp_id = list(label_resnames) * nb_sets
    label_chainids[label_chainids == ""] = "."
    list_label_asym_id = list(label_chainids) * nb_sets
    list_label_entity_id = list(entity_ids) * nb_sets
    label_resnums[label_resnums == "0"] = "."
    list_label_seq_id = list(label_resnums) * nb_sets
    icodes[icodes == ""] = "?"
    list_pdbx_PDB_ins_code = list(icodes) * nb_sets
    list_occupancy = list(occupancies) * nb_sets
    list_B_iso_or_equiv = list(bfactors) * nb_sets
    list_pdbx_formal_charge = list(charges2) * nb_sets
    list_auth_seq_id = list(resnums) * nb_sets
    list_auth_comp_id = list(resnames) * nb_sets
    list_auth_asym_id = list(chainids) * nb_sets
    list_auth_atom_id = list(atomnames) * nb_sets

    list_Cartn_x = []
    list_Cartn_y = []
    list_Cartn_z = []
    list_pdbx_PDB_model_num = []
    for m, coords in enumerate(coordsets):
        list_Cartn_x += list(coords[:,0])
        list_Cartn_y += list(coords[:,1])
        list_Cartn_z += list(coords[:,2])
        list_pdbx_PDB_model_num += [m+1] * nb_atoms


    #Get the column widths
    max_group_PDB = len(max(list_group_PDB, key=len))
    max_id = len(str(max(list_id)))
    max_type_symbol = len(max(list_type_symbol, key=len))
    max_label_atom_id = len(max(list_label_atom_id, key=len))
    max_label_alt_id = len(max(list_label_alt_id, key=len))
    max_label_comp_id = len(max(list_label_comp_id, key=len))
    max_label_asym_id = len(max(list_label_asym_id, key=len))
    max_label_entity_id = len(str(max(list_label_entity_id)))
    max_label_seq_id = len(max(list_label_seq_id, key=len))
    max_pdbx_PDB_ins_code = len(max(list_pdbx_PDB_ins_code, key=len))
    max_Cartn_x = max([len(str(x).split(".")[0]) for x in list_Cartn_x]) + 4
    max_Cartn_y = max([len(str(x).split(".")[0]) for x in list_Cartn_y]) + 4
    max_Cartn_z = max([len(str(x).split(".")[0]) for x in list_Cartn_z]) + 4
    max_occupancy = 4
    max_B_iso_or_equiv = len(str(max(list_B_iso_or_equiv)).split(".")[0]) + 3
    max_pdbx_formal_charge = len(max(list_pdbx_formal_charge, key=len))
    max_auth_seq_id = len(str(max(list_auth_seq_id)))
    max_auth_comp_id = len(max(list_auth_comp_id, key=len))
    max_auth_asym_id = len(max(list_auth_asym_id, key=len))
    max_auth_atom_id = len(max(list_auth_atom_id, key=len))
    max_pdbx_PDB_model_num = len(str(max(list_pdbx_PDB_model_num)))

    write("loop_\n"
          "_atom_site.group_PDB \n"
          "_atom_site.id \n"
          "_atom_site.type_symbol \n"
          "_atom_site.label_atom_id \n"
          "_atom_site.label_alt_id \n"
          "_atom_site.label_comp_id \n"
          "_atom_site.label_asym_id \n"
          "_atom_site.label_entity_id \n"
          "_atom_site.label_seq_id \n"
          "_atom_site.pdbx_PDB_ins_code \n"
          "_atom_site.Cartn_x \n"
          "_atom_site.Cartn_y \n"
          "_atom_site.Cartn_z \n"
          "_atom_site.occupancy \n"
          "_atom_site.B_iso_or_equiv \n"
          "_atom_site.pdbx_formal_charge \n"
          "_atom_site.auth_seq_id \n"
          "_atom_site.auth_comp_id \n"
          "_atom_site.auth_asym_id \n"
          "_atom_site.auth_atom_id \n"
          "_atom_site.pdbx_PDB_model_num \n")

    for i, _ in enumerate(list_group_PDB):
        write(f"{list_group_PDB[i]:<{max_group_PDB}s} "
              f"{list_id[i]:<{max_id}d} "
              f"{list_type_symbol[i]:<{max_type_symbol}s} "
              f"{list_label_atom_id[i]:<{max_label_atom_id}s} "
              f"{list_label_alt_id[i]:<{max_label_alt_id}s} "
              f"{list_label_comp_id[i]:<{max_label_comp_id}s} "
              f"{list_label_asym_id[i]:<{max_label_asym_id}s} "
              f"{list_label_entity_id[i]:<{max_label_entity_id}d} "
              f"{list_label_seq_id[i]:<{max_label_seq_id}s} "
              f"{list_pdbx_PDB_ins_code[i]:<{max_pdbx_PDB_ins_code}s} "
              f"{list_Cartn_x[i]:<{max_Cartn_x}.3f} "
              f"{list_Cartn_y[i]:<{max_Cartn_y}.3f} "
              f"{list_Cartn_z[i]:<{max_Cartn_z}.3f} "
              f"{list_occupancy[i]:<{max_occupancy}.2f} "
              f"{list_B_iso_or_equiv[i]:<{max_B_iso_or_equiv}.2f} "
              f"{list_pdbx_formal_charge[i]:<{max_pdbx_formal_charge}s} "
              f"{list_auth_seq_id[i]:<{max_auth_seq_id}d} "
              f"{list_auth_comp_id[i]:<{max_auth_comp_id}s} "
              f"{list_auth_asym_id[i]:<{max_auth_asym_id}s} "
              f"{list_auth_atom_id[i]:<{max_auth_atom_id}s} "
              f"{list_pdbx_PDB_model_num[i]:<{max_pdbx_PDB_model_num}d} \n")
    write("#")

    if anisous is not None and anisous_flag:
        anisotrop_U11 = anisous[:,0]
        anisotrop_U22 = anisous[:,1]
        anisotrop_U33 = anisous[:,2]
        anisotrop_U12 = anisous[:,3]
        anisotrop_U13 = anisous[:,4]
        anisotrop_U23 = anisous[:,5]
        max_anisotrop_U11 = max([len(str(x).split(".")[0]) for x in anisotrop_U11]) + 5
        max_anisotrop_U22 = max([len(str(x).split(".")[0]) for x in anisotrop_U22]) + 5
        max_anisotrop_U33 = max([len(str(x).split(".")[0]) for x in anisotrop_U33]) + 5
        max_anisotrop_U12 = max([len(str(x).split(".")[0]) for x in anisotrop_U12]) + 5
        max_anisotrop_U13 = max([len(str(x).split(".")[0]) for x in anisotrop_U13]) + 5
        max_anisotrop_U23 = max([len(str(x).split(".")[0]) for x in anisotrop_U23]) + 5

        write("loop_\n"
              "_atom_site_anisotrop.id \n"
              "_atom_site_anisotrop.type_symbol \n"
              "_atom_site_anisotrop.pdbx_label_atom_id \n"
              "_atom_site_anisotrop.pdbx_label_alt_id \n"
              "_atom_site_anisotrop.pdbx_label_comp_id \n"
              "_atom_site_anisotrop.pdbx_label_asym_id \n"
              "_atom_site_anisotrop.pdbx_label_seq_id \n"
              "_atom_site_anisotrop.pdbx_PDB_ins_code \n"
              "_atom_site_anisotrop.U[1][1] \n"
              "_atom_site_anisotrop.U[2][2] \n"
              "_atom_site_anisotrop.U[3][3] \n"
              "_atom_site_anisotrop.U[1][2] \n"
              "_atom_site_anisotrop.U[1][3] \n"
              "_atom_site_anisotrop.U[2][3] \n"
              "_atom_site_anisotrop.pdbx_auth_seq_id \n"
              "_atom_site_anisotrop.pdbx_auth_comp_id \n"
              "_atom_site_anisotrop.pdbx_auth_asym_id \n"
              "_atom_site_anisotrop.pdbx_auth_atom_id \n")

        for i, _ in enumerate(list_group_PDB):
            if anisotrop_U11[i] != 0 and anisotrop_U22[i] != 0 and anisotrop_U33[i] != 0 and anisotrop_U12[i] != 0 and anisotrop_U13[i] != 0 and anisotrop_U23[i] != 0:
                write(f"{list_id[i]:<{max_id}d} "
                    f"{list_type_symbol[i]:<{max_type_symbol}s} "
                    f"{list_label_atom_id[i]:<{max_label_atom_id}s} "
                    f"{list_label_alt_id[i]:<{max_label_alt_id}s} "
                    f"{list_label_comp_id[i]:<{max_label_comp_id}s} "
                    f"{list_label_asym_id[i]:<{max_label_asym_id}s} "
                    f"{list_label_seq_id[i]:<{max_label_seq_id}s} "
                    f"{list_pdbx_PDB_ins_code[i]:<{max_pdbx_PDB_ins_code}s} "
                    f"{anisotrop_U11[i]:<{max_anisotrop_U11}.4f} "
                    f"{anisotrop_U22[i]:<{max_anisotrop_U22}.4f} "
                    f"{anisotrop_U33[i]:<{max_anisotrop_U33}.4f} "
                    f"{anisotrop_U12[i]:<{max_anisotrop_U12}.4f} "
                    f"{anisotrop_U13[i]:<{max_anisotrop_U13}.4f} "
                    f"{anisotrop_U23[i]:<{max_anisotrop_U23}.4f} "
                    f"{list_auth_seq_id[i]:<{max_auth_seq_id}d} "
                    f"{list_auth_comp_id[i]:<{max_auth_comp_id}s} "
                    f"{list_auth_asym_id[i]:<{max_auth_asym_id}s} "
                    f"{list_auth_atom_id[i]:<{max_auth_atom_id}s} \n")
        write("#")


writeMMCIFStream.__doc__ += _writeMMCIFdoc


writeMMCIF.__doc__ += _writeMMCIFdoc + """
    :arg autoext: when not present, append extension :file:`.cif` to *filename*
"""


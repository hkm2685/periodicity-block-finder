import numpy as np
from fractions import Fraction
from itertools import combinations, product
from math import gcd
from functools import reduce

from lib_temper.temper import cokernel, defactored_hnf
from lib_temper.optimize import lstsq
from lib_temper.subgroup import p_limit
from lib_temper.interval import factors

def prime_factors(n):
    factors = []
    for d in range(2, int(n**0.5) + 1): 
        while n % d == 0: factors.append(d); n //= d
    return factors + ([n] if n > 1 else [])

def parse_subgroup(subgroup_str: str) -> list[Fraction]:
    if not subgroup_str.strip(): return []
    return [Fraction(s) for s in subgroup_str.replace('.', ' ').split()]

def parse_commas(comma_str: str, subgroup: list[Fraction] = [Fraction(p) for p in p_limit(97)]) -> list[Fraction]:
    comma_str = comma_str.strip()
    if not comma_str:
        return []

    if comma_str.startswith('['):

        try:
            cleaned_str = comma_str.replace('[', '').replace(']', '').replace(',', ' ').replace('>', ' >')
            monzo_parts = cleaned_str.split('>')
            monzo_strs = [part.strip() for part in monzo_parts if part.strip()]
            
            commas = []
            for s in monzo_strs:
                monzo = np.fromstring(s, dtype=int, sep=' ')
                if len(monzo) > len(subgroup):
                    raise ValueError(f"Monzo length {len(monzo)} does not match subgroup length {len(subgroup)}")
                elif len(monzo) < len(subgroup):
                    monzo = monzo[:len(subgroup)]
                
                comma = reduce(lambda x, y: x * y, [subgroup[i]**monzo[i] for i in range(len(monzo))], 1)
                commas.append(comma)
            return commas
            
        except ValueError as e:
            raise ValueError(f"Failed to parse monzo string: {comma_str}. Error: {e}")
    else:
        return [Fraction(s) for s in comma_str.split(', ')]

def _setup_temperament_analysis(comma_str: str, subgroup_str: str):
    """Prepares temperament data for analysis."""

    if not subgroup_str:
        parsed_commas = parse_commas(comma_str)
        subgroup_str = ".".join([str(f) for f in sorted(set(
            f for c in parsed_commas for f in prime_factors(c.denominator) + prime_factors(c.numerator)))])
        subgroup = parse_subgroup(subgroup_str)
    else:
        subgroup = parse_subgroup(subgroup_str)
        parsed_commas = parse_commas(comma_str, subgroup)

    prime_subgroup = sorted(list(set(f for s in subgroup for f in prime_factors(s.numerator) + prime_factors(s.denominator))))

    if not len(parsed_commas):
        mapping = np.eye(len(prime_subgroup))
    else:
        mapping = defactored_hnf(cokernel(np.hstack([factors(c, prime_subgroup) for c in parsed_commas])))
    
    solution, _ = lstsq((mapping, prime_subgroup), weight="tenney")
    
    return parsed_commas, prime_subgroup, mapping, solution, subgroup_str

def fokker_block(tmonzos: list[tuple], temperament_comma_str: str, subgroup_str: str="", 
                 offset: tuple=(), _prime_subgroup=None, _mapping=None, _solution=None):
    """ also allows tmonzos to be a comma string """
    if _prime_subgroup is None or _mapping is None or _solution is None:
        _, prime_subgroup, mapping, solution, subgroup_str_out = _setup_temperament_analysis(temperament_comma_str, subgroup_str)
    else:
        prime_subgroup, mapping, solution = _prime_subgroup, _mapping, _solution
        subgroup_str_out = subgroup_str

    # if tmonzos was passed as a comma string, convert it here. find the commas' monzos and apply the mapping
    if isinstance(tmonzos, str):
        monzos = [factors(comma, prime_subgroup) for comma in parse_commas(tmonzos, parse_subgroup(subgroup_str_out))]
        tmonzos = [np.dot(mapping, monzo) for monzo in monzos]
        
        # forbid enfactored comma bases
        reduced_tmonzos = []
        for tmonzo in tmonzos:
            vector = tmonzo.flatten() # tmonzo is a 2D array, flatten it to get the vector
            vector_gcd = reduce(gcd, [abs(x) for x in vector if x != 0])
            
            if vector_gcd > 1: print(f"Warning: enfactored comma bases must be expressed as tmonzos.")
            reduced_vector = vector // vector_gcd
            reduced_tmonzos.append(tuple(reduced_vector))
        
        tmonzos = reduced_tmonzos

    # get list of tmonzos inside parallelotope. offset is left bottom corner. also get tunings
    basis_vectors = np.array([t[1:] for t in tmonzos])
    dim = basis_vectors.shape[1]
    tunings = []

    if not offset: offset = np.zeros(dim)
    else: offset = np.array(offset)

    if basis_vectors.shape[0] != dim: raise ValueError("The number of tmonzo vectors must be" 
                            "equal to the dimension of the space after the first coordinate.")

    try: basis_inv = np.linalg.inv(basis_vectors)
    except np.linalg.LinAlgError: raise ValueError("tmonzos must be linearly independent.")

    iter_min = np.floor(np.sum(np.minimum(0, basis_vectors), axis=0) + offset).astype(int)
    iter_max = np.ceil (np.sum(np.maximum(0, basis_vectors), axis=0) + offset).astype(int)
    lattice_points = []
    ranges = [range(min_c, max_c + 1) for min_c, max_c in zip(iter_min, iter_max)]
    for p_tuple in product(*ranges):
        p = np.array(p_tuple)
        alpha = (p - offset) @ basis_inv
        if np.all((alpha >= 0) & (alpha < 1)):
            lattice_points.append(p)

    points_inside = []
    if lattice_points:
        s = solution.flatten()
        if len(s) != dim + 1:
            raise ValueError(f"Number of generators {len(s)} does not match tmonzo dimension {dim+1}.")
        
        for p in lattice_points:
            val = np.dot(s[1:], p)
            c1 = 1 - np.ceil(val / s[0])
            points_inside.append(tuple(np.concatenate(([c1], p)).astype(int)))
            tunings.append((val + c1 * s[0]) * 1200)  # in cents
    
    # get epimorph val
    epimorph_val = cokernel(np.array(tmonzos).T)

    # sort points inside by size for checking strength
    equal_order = []
    zipped = sorted(zip(points_inside, tunings), key=lambda x: x[1])
    for point, _ in zipped:
        equal_order.append(np.dot(epimorph_val, point))

    return zipped, tuple(epimorph_val[0]), all(equal_order[i] == i+1 for i in range(len(equal_order)))

def find_strong_blocks(note_range: tuple[int, int], comma_str: str, subgroup_str: str="", offset: tuple=()):
    low_count, high_count = note_range
    parsed_commas, prime_subgroup, mapping, solution, subgroup_str_out = _setup_temperament_analysis(comma_str, subgroup_str)

    dim = len(prime_subgroup) - len(parsed_commas) - 1
    if dim <= 0: return []

    s = solution.flatten()

    candidate_tmonzos = set()
    exponent_ranges = [range(high_count+1)] + [range(-high_count, high_count + 1)] * (dim-1)
    for p_tuple in product(*exponent_ranges):
        p = np.array(p_tuple)

        if np.linalg.norm(p) > high_count:
            continue

        val = np.dot(s[1:], p)
        t_0 = int(np.round(-val / s[0]))

        tmonzo = np.concatenate(([t_0], p)).astype(int)
        if not np.any(tmonzo):
            continue

        candidate_tmonzos.add(tuple(tmonzo)) # includes tmonzos with common factors

    candidate_tmonzos = sorted(list(candidate_tmonzos), key=lambda x: (sum(abs(c) for c in x), x))

    strong_blocks = []
    for tmonzo_basis in combinations(candidate_tmonzos, dim):
        try:
            basis_vectors = np.array([t[1:] for t in tmonzo_basis])
            if basis_vectors.shape[0] == basis_vectors.shape[1]:
                num_notes = round(abs(np.linalg.det(basis_vectors)))
                if not (low_count <= num_notes <= high_count):
                    continue
            
            chroma_sizes = [abs(np.dot(s, np.array(t)) * 1200) for t in tmonzo_basis]
            avg_chroma_size = np.mean(chroma_sizes)
            
            # To find the longest diagonal, we check all 2^(d-1) diagonals
            longest_diagonal_len = 0
            if dim > 0:
                for i in range(2**(dim)):
                    signs = np.array([(1 if (i >> j) & 1 else -1) for j in range(dim-1)])
                    diagonal = np.sum(basis_vectors[1:] * signs[:, np.newaxis], axis=0)
                    longest_diagonal_len = max(longest_diagonal_len, np.linalg.norm(diagonal))

            result, val, is_strong = fokker_block(
                list(tmonzo_basis),
                comma_str,
                subgroup_str_out,
                offset,
                _prime_subgroup=prime_subgroup,
                _mapping=mapping,
                _solution=solution
            )
            
            if is_strong:
                strong_blocks.append({
                    'tmonzos': tmonzo_basis,
                    'notes': result,
                    'val': val,
                    'note_count': len(result),
                    'avg_chroma_size': avg_chroma_size,
                    'longest_diagonal': longest_diagonal_len,
                })
        except ValueError:
            continue
            
    return sorted(strong_blocks, key=lambda x: x['longest_diagonal'])

if __name__ == "__main__":
    # print(*fokker_block([(-2, 20)], "225/224, 1029/1024", offset=(0)), sep="\n")
    # print(*fokker_block("25/24", "225/224, 1029/1024", offset=(0)), sep="\n")
    # print(*fokker_block("81/80, 49/48", "441/440, 540/539"), sep="\n")
    # print(*fokker_block([(-9, 6)], "81/80"), sep="\n")
    # print(*fokker_block("25/24", "81/80"), sep="\n")
    # print(*fokker_block("1029/1024", "[-4 -1 2>", "2.3.7"), sep="\n")
    # print(*fokker_block("1029/1024", "[-4 -1 0 2>"), sep="\n")

    print(*find_strong_blocks((12, 20), "441/440, 540/539")[:5], sep="\n\n")
    # print(*find_strong_blocks((9, 9), "", "2.3.5"), sep="\n\n")
    # print(*find_strong_blocks((4, 26), "49/48", offset=(-2,)), sep="\n\n")

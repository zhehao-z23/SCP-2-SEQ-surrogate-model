"""Deterministic sequence composition and theoretical charge, standard library only.

The pKa numbers follow Biopython 1.85 Bio.SeqUtils.IsoelectricPoint (Bjellqvist).
The implementation is independent; pI uses an explicitly checked [0, 14] bracket
instead of Biopython's default [4.05, 12]. These are sequence-derived proxies,
not structural pKa, teacher protonation assignments, or surface potentials.
"""

from collections import Counter
import hashlib
import json
import math

VERSION = "sequence_charge_bjellqvist_v1.0.0"
AA20 = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_PH_VALUES = (3.0, 4.0, 5.0, 6.0, 7.0, 7.4, 8.0, 9.0, 10.0, 11.0)
PKA_POSITIVE = {"K": 10.0, "R": 12.0, "H": 5.98}
PKA_NEGATIVE = {"D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
NTERM_DEFAULT = 7.5
CTERM_DEFAULT = 3.55
NTERM_OVERRIDES = {"A": 7.59, "M": 7.0, "S": 6.93, "P": 8.36,
                   "T": 6.82, "V": 7.44, "E": 7.7}
CTERM_OVERRIDES = {"D": 4.55, "E": 4.75}
PI_BRACKET = (0.0, 14.0)
PI_PH_TOLERANCE = 1e-10
PI_CHARGE_TOLERANCE = 1e-8
REFERENCE_SOURCE = (
    "https://raw.githubusercontent.com/biopython/biopython/biopython-185/"
    "Bio/SeqUtils/IsoelectricPoint.py"
)


def _ph(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 14.0:
        raise ValueError("pH must be finite and within the declared [0, 14] range")
    return value


def _canonical(sequence):
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string; parse FASTA before calling")
    # Uppercasing is explicit; gaps, whitespace, terminal '*' are never removed.
    # Restrict lowercase normalization to ASCII so a Unicode letter cannot expand
    # into multiple standard residues (e.g. sharp-s -> SS).
    return "".join(chr(ord(x) - 32) if "a" <= x <= "z" else x for x in sequence)


def _standard(sequence):
    sequence = _canonical(sequence)
    if not sequence or any(x not in AA20 for x in sequence):
        raise ValueError("charge calculation requires a nonempty standard-20AA sequence")
    return sequence


def _components(counts, first, last, ph):
    positive = math.fsum(n / (1.0 + 10.0 ** (ph - PKA_POSITIVE[aa]))
                         for aa, n in counts.items() if aa in PKA_POSITIVE)
    negative = -math.fsum(n / (1.0 + 10.0 ** (PKA_NEGATIVE[aa] - ph))
                          for aa, n in counts.items() if aa in PKA_NEGATIVE)
    nterm = 1.0 / (1.0 + 10.0 ** (ph - NTERM_OVERRIDES.get(first, NTERM_DEFAULT)))
    cterm = -1.0 / (1.0 + 10.0 ** (CTERM_OVERRIDES.get(last, CTERM_DEFAULT) - ph))
    sidechain = math.fsum((positive, negative))
    return {"q_positive_sidechain": positive, "q_negative_sidechain": negative,
            "q_sidechain": sidechain, "q_nterm": nterm, "q_cterm": cterm,
            "q_free_chain": math.fsum((positive, negative, nterm, cterm))}


def charge_at_ph(sequence, ph, *, include_termini=False):
    """Return signed elementary-charge units under independent-site pKa values.

    include_termini=True adds exactly one unblocked N terminus and C terminus.
    It does not detect physical cleavage, amidation, acetylation or disulfides.
    Unsupported or empty sequences raise, rather than silently becoming neutral.
    """
    sequence = _standard(sequence)
    values = _components(Counter(sequence), sequence[0], sequence[-1], _ph(ph))
    return values["q_free_chain" if include_termini else "q_sidechain"]


def _solve_pi(counts, first, last, bracket):
    lo, hi = (_ph(x) for x in bracket)
    if lo >= hi:
        raise ValueError("pI bracket must be strictly increasing")
    initial_lo, initial_hi = lo, hi
    def q(ph):
        return _components(counts, first, last, ph)["q_free_chain"]
    qlo, qhi = q(lo), q(hi)
    result = {"value": None, "status": "root_not_bracketed",
              "residual": None, "bracket_low": initial_lo,
              "bracket_high": initial_hi, "q_low": qlo, "q_high": qhi,
              "iterations": 0}
    if not (qlo > 0.0 and qhi < 0.0):
        # Do not return a range boundary and pretend it is a root.
        return result
    for iteration in range(1, 97):
        mid = (lo + hi) / 2.0
        residual = q(mid)
        if residual > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= PI_PH_TOLERANCE and abs(residual) <= PI_CHARGE_TOLERANCE:
            result.update(value=mid, status="ok", residual=residual, iterations=iteration)
            return result
    result.update(status="root_tolerance_failed", residual=residual, iterations=96)
    return result


def isoelectric_point(sequence, *, bracket=PI_BRACKET):
    """Return a checked free-termini theoretical pI result, including residual.

    No sidechain-only pI is returned: a neutral sequence has infinitely many
    sidechain-only zeros, while a single-charge-class sequence may have no zero.
    """
    sequence = _standard(sequence)
    return _solve_pi(Counter(sequence), sequence[0], sequence[-1], bracket)


def _ph_suffix(value):
    return format(value, ".12g").replace(".", "p")


def extract_features(sequence, *, sequence_view, full_chain,
                     ph_values=DEFAULT_PH_VALUES):
    """Return one flat JSON-safe row without dropping unsupported sequences.

    sequence_view identifies C/O/T/U or a declared caller-specific view. The
    full_chain boolean must be supplied. False labels all free-chain quantities
    as hypothetical concatenated-sequence proxies; it never inserts gap termini.
    AA fractions use the entire input length, retaining unsupported residues in
    the denominator. Charge and KRDE summaries are null if any residue is not
    standard, even when some known contributions could be calculated.
    """
    if not isinstance(sequence_view, str) or not sequence_view:
        raise ValueError("sequence_view must explicitly identify the input object")
    if type(full_chain) is not bool:
        raise TypeError("full_chain must be explicitly True or False")
    normalized = _canonical(sequence)
    ph_values = tuple(_ph(x) for x in ph_values)
    suffixes = tuple(_ph_suffix(x) for x in ph_values)
    if not ph_values or len(set(suffixes)) != len(ph_values):
        raise ValueError("pH values must be nonempty and uniquely named")
    counts = Counter(normalized)
    length = len(normalized)
    unsupported = {aa: n for aa, n in sorted(counts.items()) if aa not in AA20}
    invalid = [aa for aa in unsupported if not ("A" <= aa <= "Z")]
    if not length:
        status = "unsupported_empty_sequence"
    elif invalid:
        status = "unsupported_invalid_characters"
    elif unsupported:
        status = "unsupported_nonstandard_residues"
    else:
        status = "ok"
    # Invalid characters are not biological residues: don't expose their ratios
    # as a valid composition. Letter-only U/O/X/B/Z/J rows retain partial counts.
    composition_ok = bool(length) and not invalid
    row = {
        "feature_version": VERSION,
        "sequence_view": sequence_view,
        "full_chain_declared": full_chain,
        "sequence_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "input_sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "uppercase_normalization_applied": normalized != sequence,
        "length": length,
        "log1p_length": math.log1p(length),
        "composition_status": ("ok" if status == "ok" else
                               "partial_known_counts" if composition_ok else status),
        "charge_feature_status": status,
        "unsupported_count": sum(unsupported.values()),
        "unsupported_counts_json": json.dumps(unsupported, sort_keys=True, ensure_ascii=True),
        "unsupported_fraction": sum(unsupported.values()) / length if composition_ok else None,
        "terminal_assumption": "one_unblocked_Nterm_and_one_unblocked_Cterm",
        "terminal_interpretation": ("full_sequence_free_termini_proxy" if full_chain else
                                    "hypothetical_concatenated_sequence_proxy"),
        "pi_free_chain": None,
        "pi_status": status,
        "pi_residual_charge": None,
        "pi_bracket_low": PI_BRACKET[0],
        "pi_bracket_high": PI_BRACKET[1],
        "pi_q_low": None,
        "pi_q_high": None,
        "pi_iterations": None,
    }
    for aa in AA20:
        row["count_" + aa] = counts[aa]
        row["fraction_" + aa] = counts[aa] / length if composition_ok else None
    for name in ("f_positive_KR", "f_negative_DE", "fcr_KRDE", "signed_ncpr_KRDE",
                 "abs_ncpr_KRDE", "f_ionizable_sidechains_KRHDECY"):
        row[name] = None
    for suffix in suffixes:
        for prefix in ("q_positive_sidechain", "q_negative_sidechain", "q_sidechain",
                       "q_nterm", "q_cterm", "q_free_chain", "q_sidechain_per_residue",
                       "q_free_chain_per_residue"):
            row[prefix + "_ph" + suffix] = None
    if status != "ok":
        return row
    positive = counts["K"] + counts["R"]
    negative = counts["D"] + counts["E"]
    row.update(f_positive_KR=positive / length, f_negative_DE=negative / length,
               fcr_KRDE=(positive + negative) / length,
               signed_ncpr_KRDE=(positive - negative) / length,
               abs_ncpr_KRDE=abs(positive - negative) / length,
               f_ionizable_sidechains_KRHDECY=sum(counts[x] for x in "KRHDECY") / length)
    pi = _solve_pi(counts, normalized[0], normalized[-1], PI_BRACKET)
    row.update(pi_free_chain=pi["value"], pi_status=pi["status"],
               pi_residual_charge=pi["residual"], pi_q_low=pi["q_low"],
               pi_q_high=pi["q_high"], pi_iterations=pi["iterations"])
    if pi["status"] != "ok":
        row["charge_feature_status"] = "pi_" + pi["status"]
    for ph, suffix in zip(ph_values, suffixes):
        components = _components(counts, normalized[0], normalized[-1], ph)
        components["q_sidechain_per_residue"] = components["q_sidechain"] / length
        components["q_free_chain_per_residue"] = components["q_free_chain"] / length
        for name, value in components.items():
            row[name + "_ph" + suffix] = value
    return row


def feature_contract():
    """Machine-readable model assumptions and exact feature schema."""
    return {
        "version": VERSION, "aa_order": AA20, "dtype": "Python float / float64",
        "dependencies": "Python standard library only",
        "reference": {"implementation": "Biopython 1.85 IsoelectricPoint pKa tables",
                      "source": REFERENCE_SOURCE,
                      "docs": "https://biopython.org/docs/1.85/api/Bio.SeqUtils.IsoelectricPoint.html"},
        "pka": {"positive_sidechains": dict(PKA_POSITIVE),
                "negative_sidechains": dict(PKA_NEGATIVE),
                "Nterm_default": NTERM_DEFAULT, "Cterm_default": CTERM_DEFAULT,
                "Nterm_overrides": dict(NTERM_OVERRIDES),
                "Cterm_overrides": dict(CTERM_OVERRIDES)},
        "ph_values": list(DEFAULT_PH_VALUES), "ph_allowed_range": [0.0, 14.0],
        "pi_bracket": list(PI_BRACKET), "pi_ph_tolerance": PI_PH_TOLERANCE,
        "pi_charge_tolerance": PI_CHARGE_TOLERANCE,
        "pi_solver": "bisection; require positive-low/negative-high bracket and both tolerances",
        "charge_unit": "elementary charge e; signed negative contributions are negative",
        "fraction_denominator": "full input sequence length, including nonstandard letters",
        "nonstandard_policy": "preserve row/counts; all Q/pI/KRDE aggregate values null",
        "invalid_character_policy": "preserve row/counts; all fractions/Q/pI null; no trimming",
        "free_termini_policy": "exactly one unblocked N terminus and C terminus; no gap termini",
        "limits": ["independent fixed pKa sites; no structural shifts or teacher protonation",
                   "unmodified free thiols; no disulfide/PTM/ligand correction",
                   "full_chain=False makes terminal quantities hypothetical concatenation proxies",
                   "pH is a feature query condition; not a teacher pH assertion",
                   "no learned features, environmental model, or surface electrostatics"],
        "columns": list(extract_features("ACDEFGHIKLMNPQRSTVWY", sequence_view="U",
                                         full_chain=True).keys()),
    }

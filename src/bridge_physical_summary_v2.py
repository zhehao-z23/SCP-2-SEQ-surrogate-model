"""Export bounded engineering evidence only after immutable label acceptance.

The public output contains numerical-comparison errors and resource aggregates,
not sample identities, descriptors, electrostatic targets, or private paths.
This reads the completed acceptance chain; it neither runs nor grants a gate.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import math
import statistics

METRIC_LIMITS = {'Hellinger': 0.02, 'W1_kBT_per_e': 0.1, 'CDF_sup': 0.01}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def checked(item):
    assert sha(item['path']) == item['sha256'], item['path']
    return Path(item['path'])


def distribution(values):
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    return {'count': len(values), 'min': min(values),
            'median': statistics.median(values), 'max': max(values),
            'sum': sum(values)}


def export(root, receipt_path):
    receipt = load(receipt_path)
    assert receipt['schema'] == 'qualified_AF_development_label_receipt_v1'
    assert receipt['status'] == 'accepted' and receipt['numerical_qualified'] is True
    assert receipt['role'] == 'development_only'
    assert receipt['accepted_operator_variant'] == 'base'
    assert receipt['primary_array'] == 'corrected_area_p'
    assert receipt['thresholds'] == METRIC_LIMITS
    bound = {key: receipt[key] for key in [
        'qualification', 'integrity_audit', 'independent_audit', 'ledger_snapshot',
        'geometry_validation_revision', 'geometry_variant_admissions',
        'execution_preflight', 'execution_stage_history', 'development_role_guard',
        'acceptance_code', 'label_representation', 'sequence_and_structure']}
    for item in bound.values():
        checked(item)
    qualification = load(bound['qualification']['path'])
    integrity = load(bound['integrity_audit']['path'])
    independent = load(bound['independent_audit']['path'])
    ledger = load(bound['ledger_snapshot']['path'])
    assert qualification['numerical_qualified'] is True
    assert integrity['source_export_ledger_integrity_pass'] is True
    assert integrity['all_native_field_maps_and_archive_hashes_verified'] is True
    assert integrity['sampling_integrity']['pass'] is True
    assert independent['status'] == 'complete' and independent['pass'] is True
    for audit in [integrity, independent]:
        actual = audit['source_qualification']
        assert Path(actual['path']) == Path(bound['qualification']['path'])
        assert actual['sha256'] == bound['qualification']['sha256']
    assert len(qualification['comparisons']) == len(receipt['required_axes']) == 8
    assert {row['axis'] for row in qualification['comparisons']} == set(receipt['required_axes'])
    axes = []
    for row in qualification['comparisons']:
        assert row['pass'] is True
        metrics = {key: row['metrics'][key] for key in METRIC_LIMITS}
        assert all(math.isfinite(value) and 0 <= value <= receipt['thresholds'][key]
                   for key, value in metrics.items())
        axes.append({'axis': row['axis'], 'test_variant': row['test'],
                     'reference_variant': row['reference'], 'pass': True,
                     'difference_metrics': metrics})
    attempts = ledger['attempts']
    counts = dict(collections.Counter(row['status'] for row in attempts))
    assert counts.get('running', 0) == counts.get('reserved', 0) == 0
    assert len(attempts) <= ledger['cap_including_failed_attempts'] == 120
    field_rows = integrity['field_audits']
    assert len({row['PB_attempt_id'] for row in field_rows}) == len(field_rows)
    fields = [load(checked(row['receipt'])) for row in field_rows]
    assert all(field['result']['returncode'] == 0 and not field['result']['timed_out']
               and field['result']['threads'] == 1 for field in fields)
    native_starts = [datetime.datetime.fromisoformat(field['result']['started_at_utc']) for field in fields]
    native_ends = [datetime.datetime.fromisoformat(field['result']['finished_at_utc']) for field in fields]
    native_span = (max(native_ends) - min(native_starts)).total_seconds()
    assert native_span >= 0
    further_path = root / 'records/further_complete.json'
    complete = load(further_path)
    assert complete['strict_per_instance_qualified'] is True
    assert complete['completed_child_fields'] == 68
    history = load(bound['execution_stage_history']['path'])
    starts = [0]
    for item in history['resumed_stages']:
        stage = load(checked(item))
        assert stage['status'] == 'validated_for_resume'
        starts.append(len(stage['cached_sampling_evidence']))
    assert starts == sorted(set(starts))
    stops = starts[1:] + [complete['completed_child_fields']]
    stages = [{'stage': i + 1, 'completed_fields_before': start,
               'completed_fields_added': stop - start,
               'completed_fields_cumulative': stop}
              for i, (start, stop) in enumerate(zip(starts, stops))]
    sampling = integrity['sampling_integrity']['variants']
    assert len(sampling) == 9 and all(row['pass'] is True for row in sampling)
    auxiliary_path = root / 'results' / (receipt['sample_id'] + '_decomposed_comparisons.json')
    auxiliary = load(auxiliary_path)
    assert auxiliary['changes_do_not_replace_original_required_pairs'] is True
    assert {row['axis'] for row in auxiliary['rows']} == {
        'matched_parent_spacing', 'matched_intermediate_further'}
    residuals = {
        'matched_parent_spacing': 'Matches the outer parent grid; fine box, partition count and actual local halo still differ.',
        'matched_intermediate_further': 'Uses the intermediate result as actual nested boundary data; fine spacing, local domains, halos and boundary-propagation path still differ.'}
    auxiliary_rows = []
    for row in auxiliary['rows']:
        metrics = {key: row['metrics'][key] for key in METRIC_LIMITS}
        assert all(math.isfinite(value) and value >= 0 for value in metrics.values())
        auxiliary_rows.append({'comparison': row['axis'], 'difference_metrics': metrics,
                               'descriptive_within_original_limits': row['pass'],
                               'additional_PB_calls': 0, 'replaces_required_gate': False,
                               'residual_joint_changes': residuals[row['axis']]})
    summary = {
        'schema': 'anonymous_AF_numerical_bridge_engineering_summary_v2',
        'status': 'accepted_development_only', 'qualified_coordinate_objects': 1,
        'self_executed_required_axes': 8, 'inherited_instead_of_self_executed_axes': 0,
        'thresholds': receipt['thresholds'], 'axes': axes,
        'auxiliary_comparisons': auxiliary_rows,
        'registered_comparison_scope': 'The spacing and coarse-parent pairs also change fine-box or partition geometry. Passing them demonstrates registered numerical sensitivity limits, not isolated causal effects.',
        'accepted_operator': 'base-resolution area-weighted potential distribution',
        'audit_gates': {'main_field_map_archive_integrity': True,
                        'independent_numerical_sampling_nested_geometry': True,
                        'immutable_successor_acceptance': True,
                        'original_outcome_flags_retained': True},
        'query_integrity': {
            'variants_verified': len(sampling),
            'all_queries_retained': all(row['all_queries_retained'] for row in sampling),
            'all_query_ownership_exactly_once': all(row['ownership_min'] == row['ownership_max'] == 1 for row in sampling),
            'all_eight_interpolation_corners_present': all(row['all_eight_corners_present'] for row in sampling),
            'uncorrected_queries_retain_raw_values': all(row['false_correction_retains_raw_exactly'] for row in sampling),
            'nested_child_fields': complete['completed_child_fields'],
            'nested_queries_owned_once': independent['nested_geometry']['total_queries'],
            'fallback_fraction_has_no_unregistered_veto_threshold': True},
        'execution': {
            'PB_attempt_status_counts': counts, 'PB_attempts_total': len(attempts),
            'PB_attempt_cap_including_failures': 120,
            'audited_unique_completed_fields': len(fields),
            'APBS_wall_seconds_per_completed_field': distribution([field['result']['wall_seconds'] for field in fields]),
            'APBS_peak_RSS_bytes_per_completed_field': {key: value for key, value in distribution([field['result']['peak_RSS_bytes'] for field in fields]).items() if key != 'sum'},
            'first_APBS_start_to_last_APBS_finish_wall_seconds': native_span,
            'native_observation_window_scope': 'Includes inter-field work and pauses inside this window; excludes preparation before the first solve and audits after the last solve. It is not full pipeline wall time or CPU accounting.',
            'true_CPU_accounting_available': False,
            'cost_scope': 'APBS receipts only; excludes map generation, meshing, sampling, validation and paused wall time. Sum is task-seconds, not elapsed pipeline time.',
            'native_APBS_threads_per_field': 1, 'maximum_concurrent_further_fields': 2,
            'logical_CPU_allocation': 4, 'physical_cores_in_allocation': 2,
            'worker_slot_separation_is_not_OS_exclusive': True,
            'further_retained_branch_bytes_at_completion': complete['NVMe_branch_bytes'],
            'further_branch_budget_bytes': 650 * 1024**3,
            'interruption_records': len(history.get('interrupted_stages', [])),
            'further_execution_stages': stages},
        'scope': {
            'numerical_stability_of_fixed_coordinate_and_protocol': True,
            'experimental_absolute_accuracy_established': False,
            'all_AFDB_or_long_sequence_validity_established': False,
            'protected_evaluation_or_release_training_permission_granted': False,
            'structure_source_effect_separated': False,
            'grid_comparisons_are_perfect_single_factor_interventions': False,
            'continuous_limit_error_bound_established': False,
            'old_and_new_histogram_heads_interchangeable': False,
            'existing_release_role_gate_still_required': True},
        'public_content': 'No sample identifiers, sequences, structure descriptors, target distributions, target quantiles or individual potential summaries.',
        'evidence_sha256': {key: bound[key]['sha256'] for key in [
            'qualification', 'integrity_audit', 'independent_audit',
            'acceptance_code', 'geometry_validation_revision']},
        'accepted_receipt_sha256': sha(receipt_path),
        'summary_code_sha256': sha(__file__)}
    provenance = {'schema': 'private_engineering_summary_provenance_v1',
                  'accepted_receipt': ref(receipt_path), 'bound_evidence': bound,
                  'further_completion': ref(further_path),
                  'auxiliary_comparisons': ref(auxiliary_path), 'summary_code': ref(__file__)}
    return summary, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--accepted-receipt', type=Path, required=True)
    parser.add_argument('--public-output', type=Path, required=True)
    parser.add_argument('--private-provenance', type=Path, required=True)
    args = parser.parse_args()
    assert args.public_output != args.private_provenance
    assert not args.public_output.exists() and not args.private_provenance.exists()
    summary, provenance = export(args.root, args.accepted_receipt)
    for path, result in [(args.public_output, summary), (args.private_provenance, provenance)]:
        with path.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    print(json.dumps({'public': ref(args.public_output), 'private': ref(args.private_provenance)}))


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Assertions must be enabled.')
    main()

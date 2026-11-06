param(
    [string]$SourceDirectory = $PSScriptRoot,
    [string]$Output = (Join-Path $PSScriptRoot 'bridge_shape_loss_probe_independent_review.json')
)
$ErrorActionPreference = 'Stop'

# Read-only decision audit. This does not alter the frozen plan or executed code.
function SamePrimaryPathOnBoth([bool]$Sstop, [bool]$Seval, [bool]$Tstop, [bool]$Teval) {
    return (($Sstop -and $Seval) -or ($Tstop -and $Teval))
}
function ExecutedV1Primary([bool]$Sstop, [bool]$Seval, [bool]$Tstop, [bool]$Teval) {
    return (($Sstop -or $Tstop) -and ($Seval -or $Teval))
}
function Check([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Digest([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

Check (-not (Test-Path -LiteralPath $Output)) 'Refusing to replace an existing review.'
$pins = [ordered]@{
    'bridge_shape_loss_probe_summary.json' = 'c20abcc692a6b7943d6cd6b1a3c2b161bf4a7a8e1307c0cdb8f1b6ec57b2aec5'
    'bridge_shape_loss_probe_plan.json' = '34fd332bf2779a8212392e2d6e7a9b911a3f483cbf392b6a10cc8411e8255b64'
    'bridge_shape_loss_train.py' = 'efe1fe540651a565a110e0bb042fdefb7382940c9c2234d95f513302af074b62'
    'bridge_shape_loss_execute.py' = 'a07b4e74505052f7a99c8e17f002bab3ae48a3735f546a0c8d3e49b5b46824b3'
    'bridge_shape_loss_analyze.py' = 'e2e907f94f4841f401820a4cccbc3ddd36ff195aed5090299768d0f56ec28e5a'
}
foreach ($entry in $pins.GetEnumerator()) {
    Check ((Digest (Join-Path $SourceDirectory $entry.Key)) -eq $entry.Value) ('Source identity changed: ' + $entry.Key)
}
$summary = Get-Content -Raw -LiteralPath (Join-Path $SourceDirectory 'bridge_shape_loss_probe_summary.json') | ConvertFrom-Json
$plan = Get-Content -Raw -LiteralPath (Join-Path $SourceDirectory 'bridge_shape_loss_probe_plan.json') | ConvertFrom-Json
Check ($summary.plan_sha256 -eq $pins['bridge_shape_loss_probe_plan.json']) 'Summary-plan binding differs.'
Check ($summary.analysis_code_sha256 -eq $pins['bridge_shape_loss_analyze.py']) 'Analyzer binding differs.'
Check ($plan.trainer_sha256 -eq $pins['bridge_shape_loss_train.py']) 'Trainer binding differs.'

$rows = @()
foreach ($partition in @('stop', 'evaluation')) {
    $h = @($summary.results | Where-Object { $_.loss -eq 'H2' -and $_.partition -eq $partition })
    $c = @($summary.results | Where-Object { $_.loss -eq 'CE_quarter' -and $_.partition -eq $partition })
    Check ($h.Count -eq 1 -and $c.Count -eq 1) 'Expected one aggregate per arm and partition.'
    $h = $h[0]; $c = $c[0]
    $shape = $c.strong_shape.union.Bowley_AE.group_macro -le .9 * $h.strong_shape.union.Bowley_AE.group_macro
    $tailBias = [Math]::Abs($c.metrics.negative_edge_bias.group_macro) -le .8 * [Math]::Abs($h.metrics.negative_edge_bias.group_macro)
    $tailAE = $c.metrics.negative_edge_AE.group_macro -lt $h.metrics.negative_edge_AE.group_macro
    $tail = $tailBias -and $tailAE
    $distribution = $c.metrics.H.group_macro -le 1.01 * $h.metrics.H.group_macro -and $c.metrics.CDF_MSE.group_macro -le 1.01 * $h.metrics.CDF_MSE.group_macro
    $hTP = $h.fixed_one_percent_ranking.negative.true_positive_T + $h.fixed_one_percent_ranking.positive.true_positive_T
    $cTP = $c.fixed_one_percent_ranking.negative.true_positive_T + $c.fixed_one_percent_ranking.positive.true_positive_T
    $ranking = $cTP -ge $hTP
    $pass = ($shape -or $tail) -and $distribution -and $ranking
    $record = $summary.frozen_promotion_decisions | Where-Object { $_.partition -eq $partition }
    foreach ($pair in @(@($shape, $record.rare_shape_10pct_improvement), @($tail, $record.tail_bias20pct_and_tail_AE_improvement), @($distribution, $record.H_CDF_within1pct_guardrail), @($ranking, $record.combined_fixed_budget_TP_not_lower), @($pass, $record.pass))) {
        Check ($pair[0] -eq $pair[1]) 'Aggregate recomputation differs from executed decision.'
    }
    $rows += [ordered]@{
        partition = $partition
        strong_shape_Bowley_AE_change_percent = 100 * ($c.strong_shape.union.Bowley_AE.group_macro / $h.strong_shape.union.Bowley_AE.group_macro - 1)
        negative_tail_bias_magnitude_reduction_percent = 100 * (1 - [Math]::Abs($c.metrics.negative_edge_bias.group_macro) / [Math]::Abs($h.metrics.negative_edge_bias.group_macro))
        negative_tail_AE_change = $c.metrics.negative_edge_AE.group_macro - $h.metrics.negative_edge_AE.group_macro
        H_change_percent = 100 * ($c.metrics.H.group_macro / $h.metrics.H.group_macro - 1)
        CDF_MSE_change_percent = 100 * ($c.metrics.CDF_MSE.group_macro / $h.metrics.CDF_MSE.group_macro - 1)
        strong_shape_10pct_pass = $shape
        tail_bias_20pct_pass = $tailBias
        tail_AE_lower = $tailAE
        tail_combined_pass = $tail
        distribution_guardrail_pass = $distribution
        fixed_budget_TP = @($hTP, $cTP)
        ranking_guardrail_pass = $ranking
        executed_partition_pass = $pass
        strong_shape_support_rows = $h.strong_shape.union.Bowley_AE.support.rows
        strong_shape_support_groups = $h.strong_shape.union.Bowley_AE.support.groups
    }
}
$samePath = SamePrimaryPathOnBoth $rows[0].strong_shape_10pct_pass $rows[1].strong_shape_10pct_pass $rows[0].tail_combined_pass $rows[1].tail_combined_pass
$executedPrimary = ExecutedV1Primary $rows[0].strong_shape_10pct_pass $rows[1].strong_shape_10pct_pass $rows[0].tail_combined_pass $rows[1].tail_combined_pass
$guards = $rows[0].distribution_guardrail_pass -and $rows[1].distribution_guardrail_pass -and $rows[0].ranking_guardrail_pass -and $rows[1].ranking_guardrail_pass
$v1Promotion = $executedPrimary -and $guards
$strictPromotion = $samePath -and $guards
Check ($v1Promotion -eq $summary.promote_to_one_paired_confirmation) 'Overall v1 decision differs.'
Check (-not $v1Promotion -and -not $strictPromotion) 'Expected this registered probe not to promote.'

$truthTable = @()
foreach ($mask in 0..15) {
    $ss = [bool]($mask -band 1); $se = [bool]($mask -band 2)
    $ts = [bool]($mask -band 4); $te = [bool]($mask -band 8)
    $old = ExecutedV1Primary $ss $se $ts $te
    $strict = SamePrimaryPathOnBoth $ss $se $ts $te
    $truthTable += [ordered]@{ Sstop=$ss; Seval=$se; Tstop=$ts; Teval=$te; executed_v1=$old; same_path_both=$strict }
}
Check (@($truthTable | Where-Object { $_.executed_v1 -ne $_.same_path_both }).Count -eq 2) 'Unexpected logical truth table.'

$trajectories = @($summary.trajectory_diagnostics | ForEach-Object {
    Check ($_.mean_clip_fraction -eq 0) 'Observed clipping is not zero.'
    [ordered]@{ loss=$_.loss; selected_epoch=$_.selected_epoch; best_at_budget_endpoint=$_.best_at_final_epoch; last10_minus_previous10_stop_CE=$_.last10_minus_previous10_stop_CE; last10_minus_previous10_stop_H=$_.last10_minus_previous10_stop_H; recorded_mean_clip_fraction=$_.mean_clip_fraction; recorded_stop_CE_floor_count_max=$_.CE_floor_count_max }
})
$result = [ordered]@{
    schema = 'anonymous_independent_fixed_path_loss_probe_review_v1'
    status = 'no_promotion_confirmed_with_executed_v1_scope_ambiguity_disclosed'
    provenance = $pins
    audit_timing = 'Post-execution read-only decision audit; not a preregistration or alteration of the original plan/analyzer.'
    scope = 'Aggregate summary and frozen local source review, plus two small execution-control receipts; no raw samples, prediction arrays, checkpoints, protected targets or new fits.'
    recomputed_partitions = $rows
    promotion = [ordered]@{
        executed_v1_primary = $executedPrimary; same_primary_path_on_both = $samePath; all_guardrails = $guards
        executed_v1_final = $v1Promotion; conservative_same_path_final = $strictPromotion
        executed_expression = '(Sstop OR Tstop) AND (Seval OR Teval)'
        conservative_reference_expression = '(Sstop AND Seval) OR (Tstop AND Teval)'
        plan_scope_note = 'The plan wording on both partitions can be read as requiring the same improvement route on both. V1 admits mixed-route successes; this audit makes the more conservative interpretation explicit without rewriting the frozen record. Both interpretations reject this actual probe.'
        truth_table_assumes_guardrails_pass = $truthTable
    }
    checkpoint_selection_review = [ordered]@{
        epoch0_eligible = $true; selected_using_stop_H_only = $true; min_delta = 0.00001
        evaluation_used_to_choose_epoch_or_LR = $false
        evidence = 'Trainer computes score from va stop group-macro H, updates best only at score < best - 1e-5, completes the frozen 134-epoch LR path, restores selected state, and only then predicts te. Scalers use tr only. The later evaluation-based promotion decision is exposed development use, not epoch selection or a blind final test.'
    }
    controls = [ordered]@{
        H2_replay_receipt_sha256 = $summary.control_replay_sha256
        paired_path_receipt_sha256 = $summary.validity_receipt_sha256
        small_receipts_read_and_hashes_matched = $true
        H2_replay_max_reported_differences = 0
        matched_epoch_batch_and_RNG_checks = 134
        new_pair_initial_model_and_RNG_equal = $true
        historical_initial_state_hash_available = $false
        control_arrays_independently_recomputed_in_this_audit = $false
    }
    trajectories = $trajectories
    interpretation_limits = @(
        'CE/4 best is epoch 134, the fixed budget endpoint, with recent stop CE and H still improving. Convergence or CE optimality is not established.'
        'Both recorded clip fractions are zero; there is no evidence to attribute the observed difference to clipping.'
        'This is one exposed fold/seed under an H2-derived schedule and H-based checkpoint selection. Loss replacement also changes gradients and AdamW trajectories; it does not isolate an asymptotic Bayes-center mechanism.'
        'The shape sets and retrieval counts are small; frozen effect thresholds are engineering rules, not significance tests. The tiny stop-tail AE reduction is only a strict numerical improvement, not evidence of a meaningful practical gain.'
        'H2 remains the current engineering baseline after this failed promotion gate. This does not show CE is useless, prove H2 caused attenuation, or establish performance on new-C or AF targets.'
    )
    recommendation = 'Do not promote or tune CE weight/LR/threshold to rescue this probe under the failed frozen rule. Preserve original execution and this post-execution clarification.'
    review_code_sha256 = Digest $PSCommandPath
}
[IO.File]::WriteAllText($Output, (($result | ConvertTo-Json -Depth 12) + "`n"), [Text.UTF8Encoding]::new($false))
[pscustomobject]@{ status=$result.status; review_sha256=(Digest $Output); executed_v1_promote=$v1Promotion; conservative_promote=$strictPromotion; truth_table_rows=$truthTable.Count } | ConvertTo-Json

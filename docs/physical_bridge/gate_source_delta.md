# Gate source update and actual execution binding

The main auditor changed from `fc11d73db374c21feea2ca5ebe3966a76d11f48f0ed95709e2f720c0c94dccd1` to `d4b73e4bad41e73aadfe8cc194ce1ffa284aa76642d91b9a08bef04f07239abd`; the acceptor changed from `146bb308ddd7f73205b20a120940f53c8c6437f2604b44fa14bf088c34389620` to `2133bf83598702534c03c0e15fb14ce1596466383981f31ba2ada49f9cf64aa2`. Old prepared sources are preserved privately.

Only the auditor's `audit()` and acceptor's `accept()` functions changed. The heavy auditor now binds the normalized PQR hash, exact bindings object, canonical ledger input hash and four original APBS log files. Acceptance requires those four checks and rehashes the exact two source files and four logs under each native working directory. The canonical JSON serialization matches the existing resume verifier.

Hellinger <= 0.02, W1 <= 0.1 kBT/e and CDF sup <= 0.01, all eight axis pairs, the base `corrected_area_p` target and the existing sampling rules remain unchanged. There is no new fallback fraction, solver residual or warning threshold. The independent prepared gate `de6cb5b47b8684f3f2400d7f4299bc7eaa63a273555ad4db68e92bcd75eb5772` is compatible with the additive main-audit fields. It binds the main result; it does not separately redo these four added source/log checks.

The prepared anonymous exporter `2c9089e847e2760f2e94a2be0239b0fbb0b34c8225e633de7c9f6c6630bfa607` takes only the three named finite scalar metrics from both required and auxiliary comparisons. Complete diagnostic objects, target vectors and case-level references remain private. Gate/resource/stage aggregates have separate explicit fields.

These are source review and preparation facts. Final audit, independent audit, acceptance and export must each bind the code actually executed; neither the old review nor this draft is an actual physical acceptance receipt.

After this limited source review, the new main auditor, independent gate and acceptor actually passed. See `source_archive_provenance.json` for their final result hashes. The initial v1 exporter executed successfully; the final v2 only changes its schema identifier and omits the per-field peak-RSS sum. No gate, numerical comparison or target was changed.

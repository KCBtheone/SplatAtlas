# SplatAtlas measurement release

This release indexes the 930 evaluated method-scene states and links each state to its recorded native execution configuration and image scores.

- [State index](states.jsonl) and [execution configurations](execution_configs.jsonl)
- [Image scores](image_scores.csv) and [per-view scores](image_scores_by_view.csv.gz)
- [Native primitive summaries](native_attributes.csv) and [fixed-state deletion responses](fixed_state_deletion_response.csv)
- Order agreement: [k=4 scene table](k4_world_aligned_scene_results.csv), [k=8 scene table](k8_world_aligned_scene_results.csv), and [k=16 scene table](k16_world_aligned_scene_results.csv). Corresponding summary and method-pair CSV files use the same names with the suffixes summary and pair_results.

The index distinguishes the selected models with public archive entries from the selected models stored on the project server. The archived assets are identified by their member paths and byte ranges. Training log entries identify recorded files in the corresponding archive. A native PLY and any companion weights support model loading, while training resumption additionally requires the optimizer state and training configuration.

Download states.jsonl and [restore_state.py](restore_state.py) into the same directory. To retrieve the indexed model and camera files for a state, run:

    python restore_state.py 2dgs_Chair ./2dgs_Chair

Use the corresponding original dataset and native implementation. The [SplatAtlas wrapper templates](https://huggingface.co/datasets/KCBtheone/splatatlas-core/tree/main/wrapper_templates) provide the method interface. The [run archive](https://huggingface.co/datasets/KCBtheone/splatatlas-runs) retains the original archive organization.

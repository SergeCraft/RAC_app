export PYTHONPATH=$PYTHONPATH:/home/sal/anaconda3/envs/tfe/lib/python3.6/site-packages/tensorflow/models/research:/home/sal/anaconda3/envs/tfe/lib/python3.6/site-packages/tensorflow/models/research/slim
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/pipeline.config --trained_checkpoint_prefix training/model.ckpt-99672 --output_directory mobileNet_graph

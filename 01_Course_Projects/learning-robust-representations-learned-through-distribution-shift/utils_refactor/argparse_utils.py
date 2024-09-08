import argparse
import json
import os
import uuid


class ArgumentParserWrapper(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--cl_strat', type=str, default="Naive",
                          help='can be either Naive or AGEM')
        self.add_argument('--folder', type=str, default="output",
                          help='Path to the folder the data is downloaded to.')
        self.add_argument('--cl_repeats', type=int, default=1,
                          help='Number of cl_repeats')
        self.add_argument('--hidden-size', type=int, default=64,
                          help='Number of channels for each convolutional layer (default: 64).')
        self.add_argument('--num_images', type=int, default=20000,
                          help='number of samples for dataset generation')
        self.add_argument('--max_num_chars', type=int, default=3,
                          help='max_num_chars for dataset generation')
        self.add_argument('--adapt_steps', type=int, default=500,
                          help='adapt_steps')
        self.add_argument('--output-folder', type=str, default=None,
                          help='Path to the output folder for saving the model (optional).')
        self.add_argument('--buffer_path', type=str, default="MNIST_single_embeddings.pt",
                          help='Path to the buffer path')
        self.add_argument('--buffer_size', type=int, default=500,
                          help="sample size of retrieval buffer presented to the model at each iteration")
        self.add_argument('--sampled_retrieval_buffer', action='store_true')
        self.add_argument('--multihot_labels', action='store_true')
        self.add_argument('--conv_preprocess', type=str, default="none", help='none or cnn or cnn_patched')
        self.add_argument('--no_scheduling', action='store_true')
        self.add_argument('--ssl_model', type=str, default="simclr.pt",
                          help='ssl model')
        self.add_argument('--encoder', type=str, default="transformer",
                          help='architecture: resnet18 or transformer')
        self.add_argument('--name', type=str, default="default_model",
                          help='run name')
        self.add_argument('--batch_size', type=int, default=16,
                          help='Number of tasks in a mini-batch of tasks (default: 16).')
        self.add_argument('--eval_batches', type=int, default=5,
                          help='number of evaluation batches')
        self.add_argument('--epochs', type=int, default=200,
                          help='epochs to train.')
        self.add_argument('--train_epochs', type=int, default=100,
                          help='epochs to train.')
        self.add_argument('--eval_every_steps', type=int, default=1000,
                          help='log validation every N epochs')
        self.add_argument('--num-workers', type=int, default=1,
                          help='Number of workers for data loading (default: 1).')
        self.add_argument('--retrieval_iterations', type=int, default=3,
                          help='Number retrieval iterations')
        self.add_argument('--num_slots', type=int, default=3,
                          help='Number of slots')
        self.add_argument('--num_heads', type=int, default=1,
                          help='Number of heads for attention')
        self.add_argument('--slot_dim', type=int, default=32,
                          help='Dimension of slots')
        self.add_argument('--seed', type=int, default=42,
                          help='random seed')
        self.add_argument('--pool', type=str, default="concat",
                          help='pooling of slots. can bei either sum or concat')
        self.add_argument('--slot_init', type=str, default="learned",
                          help='learned or sampled')
        self.add_argument('--dataset_dir', type=str, default="multimnist_23",
                          help='dir_to_dataset')
        self.add_argument('--dataset_name', type=str, default="SplitMNIST",
                          help='CL dataset name')
        self.add_argument('--strategy_name', type=str, default="Naive",
                          help='CL strategy name')
        self.add_argument('--root_dir', type=str, default="dataset/",
                          help='root dir of datasets')

        self.add_argument('--vq_baseline', type=str, default="false")
        self.add_argument('--use_full_dataset_encoder', type=str, default="false")
        self.add_argument('--add_distance_to_values', type=str, default="false")
        self.add_argument('--accept_image_fmap', type=str, default="false")
        self.add_argument('--shuffle_cl_episode', type=str, default="true")
        self.add_argument('--weight_key_visits', type=str, default="false")
        self.add_argument('--init_mode', type=str, default="kmeans", help='kmeans or uniform')
        self.add_argument('--sa_before_pooled_attention', type=str, default="true")
        self.add_argument('--learnable_keys', type=str, default="true")
        self.add_argument('--add_positional_encoding', type=str, default="true")
        self.add_argument('--frozen_encoder', type=str, default="true")
        self.add_argument('--no-retrieval', action='store_true')
        self.add_argument('--overfit-batch', action='store_true')
        self.add_argument('--concat_fetched_values', action='store_true')
        self.add_argument('--randomize_reps', type=str, default="false")
        self.add_argument('--init_latents', type=str, default="learned", help="sampled or learned")
        self.add_argument('--retrieval_corruption', type=str, default="none",
                          help="options are none, zeros, randn, same_mean_std")
        self.add_argument('--v_as_ground_truth', action='store_true')
        self.add_argument('--bottleneck_mode', type=str, default="key_multi_values",
                          help='key_multi_values or key_values_per_task or vq_baseline')
        self.add_argument('--pretrain_data', type=str, default="same",
                          help='pretrain_data')
        self.add_argument('--weighted_values', type=str, default="false",
                          help='false or true')
        self.add_argument('--use_extra_mlp_for_retrieval', action='store_true')
        self.add_argument('--shuffle_in_retrieval_set', action='store_true')
        self.add_argument('--perceiver', action='store_true')
        self.add_argument('--no_per_class_acc', action='store_true')
        self.add_argument('--no_labels_from_reps', action='store_true')
        self.add_argument('--few_shot_phase1_training', action='store_true')
        self.add_argument('--keys_from_reps', action='store_true')
        self.add_argument('--baseline', action='store_true')

        self.add_argument('--pretrain_layer', type=int, default=2,
                          help='pretrained_encoder_layer')
        self.add_argument('--cl_epochs', type=int, default=4,
                          help='cl_epochs')
        self.add_argument('--depth', type=int, default=6,
                          help='perceiver depth')
        self.add_argument('--num_books', type=int, default=1,
                          help='discrete codes num books')
        self.add_argument('--eval_with_gaussian_noise', type=float, default=0.0,
                          help='eval_with_gaussian_noise')
        self.add_argument('--n_experiences', type=int, default=10,
                          help='n_experiences')
        self.add_argument('--input_channels', type=int, default=3,
                          help='cnn input_channels')
        self.add_argument('--num_latents', type=int, default=10,
                          help='num latents perceiver')
        self.add_argument('--latent_dim', type=int, default=10,
                          help='perceiver latent dim')
        self.add_argument('--cross_dim_head', type=int, default=64,
                          help='perceiver cross_dim_head')
        self.add_argument('--cross_heads', type=int, default=1,
                          help='perceiver cross_heads')

        self.add_argument('--dim_key', type=int, default=32,
                          help='dim_key')
        self.add_argument('--dim_value', type=int, default=32,
                          help='dim_value')
        self.add_argument('--num_pairs', type=int, default=800,
                          help='num_pairs')
        self.add_argument('--num_queries', type=int, default=16,
                          help='num_queries')
        self.add_argument('--init_epochs', type=int, default=1,
                          help='init_epochs')

        self.add_argument('--latent_heads', type=int, default=8,
                          help='perceiver latent_heads')
        self.add_argument('--num_object_predictions', type=int, default=3,
                          help='perceiver num_object_predictions')
        self.add_argument('--num_freq_bands', type=int, default=6,
                          help='perceiver num_freq_bands')
        self.add_argument('--num_retrieval_attentions', type=int, default=1,
                          help='perceiver num_retrieval_attentions')
        self.add_argument('--max_freq', type=int, default=10,
                          help='perceiver max freq')
        self.add_argument('--sampled_buffer_size', type=int, default=7,
                          help='perceiver sampled_buffer_size')
        self.add_argument('--topk', type=int, default=10,
                          help='perceiver topk')

        self.add_argument('--agem_patterns_per_exp', type=int, default=250,
                          help='agem_patterns_per_exp')
        self.add_argument('--agem_sample_size', type=int, default=1300,
                          help='agem_sample_size')
        self.add_argument('--attn_dropout', type=float, default=0.0,
                          help='perceiver attn_dropout')
        self.add_argument('--ewc_lambda', type=float, default=0.1,
                          help='ewc lambda')
        self.add_argument('--retrieval_query_dropout', type=float, default=0.2,
                          help='retrieval query dropout')
        self.add_argument('--image_noise', type=float, default=0.1,
                          help='perceiver image_noise')
        self.add_argument('--learning_rate', type=float, default=3e-4,
                          help='learning_rate')
        self.add_argument('--imb_factor', type=float, default=0.01,
                          help='imb_factor')
        self.add_argument('--weight_decay', type=float, default=0.0,
                          help='weight_decay')
        self.add_argument('--commitment_weight', type=float, default=0.25,
                          help='commitment_weight')
        self.add_argument('--decay', type=float, default=0.8,
                          help='decay')
        self.add_argument('--threshold_factor', type=float, default=0.1,
                          help='threshold_factor')
        self.add_argument('--ff_dropout', type=float, default=0.0,
                          help='perceiver ff_dropout')
        self.add_argument('--buffer_model_reps_dim', type=int, default=10,
                          help='perceiver buffer_model_reps_dim')
        self.add_argument('--buffer_model_path', type=str, default=None,
                          help='perceiver buffer_model_path')
        self.add_argument('--random_projection', type=str, default="false",
                          help='random_projection')
        self.add_argument('--add_randn', type=str, default="false",
                          help='perceiver add_randn true or false')
        self.add_argument('--add_distances', type=str, default="false",
                          help='perceiver add_distances true or false')
        self.add_argument('--minimize_only_closest', type=str, default="false",
                          help='minimize_only_closest')
        self.add_argument('--cosine_weight', type=float, default=0.0,
                          help='cosine_weight for loss')
        self.add_argument('--retrieval_last', type=str, default="false",
                          help='perceiver retrieval_last true or false')
        self.add_argument('--pred_from_retrieval', type=str, default="false",
                          help='perceiver pred_from_retrieval')
        self.add_argument('--attn_retrieval', type=str, default="reps_for_k",
                          help='perceiver attn_retrieval')
        self.add_argument('--retrieval_access', type=str, default="sampled",
                          help='perceiver retrieval_access')
        self.add_argument('--extra_dataset_paths', type=str, nargs='+',
                          default=["multimnist4tiles112_112_1/test",
                                   "multimnist4tiles112_112_2/test",
                                   "multimnist4tiles112_112_3/test",
                                   "multimnist4tiles112_112_4/test"], help='extra datasets')
        self.add_argument('--test_dataset_paths', type=str, nargs='+',
                          default=["multimnist4tiles112_112_1",
                                   "multimnist4tiles112_112_2",
                                   "multimnist4tiles112_112_3",
                                   "multimnist4tiles112_112_4"], help='test datasets')
        self.add_argument('--train_dataset_paths', type=str, nargs='+',
                          default=["multimnist4tiles112_112_1",
                                   "multimnist4tiles112_112_2",
                                   "multimnist4tiles112_112_3"], help='extra datasets')
        self.add_argument('--buffer_paths_test', type=str, nargs='+', default=["MNIST_single_embeddings.pt"],
                          help='Path to the buffer path for test sets')
        self.add_argument('--viz_folder', type=str, default="viz_output",
                          help='Path to the folder the data is downloaded to.')
        self.add_argument('--decoder_size', type=str, default="two-layer",
                          help='decoder_size: either one-layer or two-layer')
        self.add_argument('--optimizer', type=str, default="Adam",
                          help='Adam or SGD')
        self.add_argument('--method', type=str, default="ours",
                          help='ours or vq_tune_single_layer or vq_tune_full_decoder')
        self.add_argument('--config', type=str, default=None,
                          help='config path of respective model')
        self.add_argument('--image_size', type=int, default=28,
                          help='size of input images')
        self.add_argument('--train_fraction', type=float, default=0.8,
                          help='train_set fraction')

        ## Buffer generation
        self.add_argument('--buffer_name', type=str, default="buffer_B28_200_per_digit",
                          help='file_name')
        self.add_argument('--buffer_samples_per_class', type=int, default=200,
                          help='samples per class')

    def parse(self):
        #if os.environ.get("https_proxy"):
        #os.environ["HTTPS_PROXY"] = "http://proxy:8080" #os.environ["https_proxy"]
        args = super().parse_args()

        args.shuffle_cl_episode = True if args.shuffle_cl_episode == "true" else False
        args.vq_baseline = True if args.vq_baseline == "true" else False
        args.random_projection = True if args.random_projection == "true" else False
        args.add_distance_to_values = True if args.add_distance_to_values == "true" else False
        args.accept_image_fmap = True if args.accept_image_fmap == "true" else False
        args.frozen_encoder = True if args.frozen_encoder == "true" else False
        args.weight_key_visits = True if args.weight_key_visits == "true" else False
        args.learnable_keys = True if args.learnable_keys == "true" else False
        args.add_positional_encoding = True if args.add_positional_encoding == "true" else False
        args.sa_before_pooled_attention = True if args.sa_before_pooled_attention == "true" else False
        args.use_full_dataset_encoder = True if args.use_full_dataset_encoder == "true" else False
        args.weighted_values = True if args.weighted_values == "true" else False
        args.minimize_only_closest = True if args.minimize_only_closest == "true" else False
        args.add_randn = True if args.add_randn == "true" else False
        args.add_distances = True if args.add_distances == "true" else False
        args.retrieval_last = True if args.retrieval_last == "true" else False
        args.pred_from_retrieval = True if args.pred_from_retrieval == "true" else False
        args.identifier = str(uuid.uuid4())
        return args

    def parse_with_config(self):
        if os.environ.get("https_proxy"):
            os.environ["HTTPS_PROXY"] = os.environ["https_proxy"]
        cl_args = super().parse_args()

        if cl_args.config is None:
            raise Exception("config path missing")
        else:
            with open(cl_args.config, 'r') as f:
                configs_dict = json.load(f)
        args_dict = vars(cl_args)
        args_dict.update(configs_dict)
        args = argparse.Namespace(**args_dict)
        return args

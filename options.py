import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--data_split', type=str, default='train', choices=['train', 'val', 'test'],
                    help='which manifest split to use for dataset loading.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--degradation_size', type=int, default=8192,
                    help='target training sample size per degradation (denoise per level uses size/2).')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                    help='gradient accumulation steps to reach larger effective batch size.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/home/huhao/adv_ir/dataset/noise/mix/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/home/huhao/adv_ir/dataset/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/home/huhao/adv_ir/dataset/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--resume_ckpt", type=str, default="",
                    help="path to a checkpoint file to resume training from")
parser.add_argument("--auto_resume", action="store_true",
                    help="resume from latest checkpoint under ckpt_dir when resume_ckpt is not provided")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")
parser.add_argument("--p_target_m_target", type=float, default=0.1,
                    help="probability for M(target)-target replacement branch in train_expairs")
parser.add_argument("--p_target_mm_input", type=float, default=0.1,
                    help="probability for M(M(input))-target replacement branch in train_expairs")

# model architecture
parser.add_argument(
    "--model_arch",
    type=str,
    default="promptir",
    choices=["nafnet", "promptir"],
    help="network architecture: default 'promptir', optional 'nafnet'",
)
parser.add_argument("--inp_channels", type=int, default=3, help="input image channels")
parser.add_argument("--out_channels", type=int, default=3, help="output image channels")
parser.add_argument("--naf_width", type=int, default=32, help="NAF width")
parser.add_argument("--naf_middle_blk_num", type=int, default=1, help="NAF middle block count")
parser.add_argument(
    "--naf_enc_blk_nums",
    type=int,
    nargs="+",
    default=[1, 1, 1, 28],
    help="NAF encoder block counts per stage",
)
parser.add_argument(
    "--naf_dec_blk_nums",
    type=int,
    nargs="+",
    default=[1, 1, 1, 1],
    help="NAF decoder block counts per stage",
)
parser.add_argument("--naf_dw_expand", type=int, default=2, help="NAF depthwise expansion ratio")
parser.add_argument("--naf_ffn_expand", type=int, default=2, help="NAF FFN expansion ratio")
parser.add_argument("--naf_dropout", type=float, default=0.0, help="NAF dropout rate")

# adversarial mix training
parser.add_argument("--adv_enable", action="store_true",
                    help="enable mixed adversarial sample training")
parser.add_argument("--adv_ratio", type=float, default=0.5,
                    help="adversarial sample ratio relative to base training set size")
parser.add_argument("--adv_resample_epochs", type=int, default=8,
                    help="resample adversarial sample pool every N epochs")
parser.add_argument("--adv_samples_per_resample", type=int, default=0,
                    help="override adversarial sample count per resample; 0 means auto by adv_ratio")
parser.add_argument("--adv_aug_k", type=int, default=1,
                    help="for haze-only adversarial generation: build n/k adversarial samples then use simple augmentations to expand to n")
parser.add_argument("--adv_steps1", type=int, default=2,
                    help="PromptIR-call steps for adversarial generation")
parser.add_argument("--adv_steps2", type=int, default=2,
                    help="inner optimization steps per fixed PromptIR gradient")
parser.add_argument("--adv_step_size", type=float, default=3e-2,
                    help="optimizer step size for adversarial degradation parameters")
parser.add_argument("--adv_lambda_reg", type=float, default=0.05,
                    help="regularization weight for adversarial degradation generation (effective value in attack is this value * 1e4)")
parser.add_argument("--adv_rain_topk", type=int, default=4,
                    help="top-k branches used in rain degradation while generating adversarial samples")
parser.add_argument("--adv_promptir_patch_size", type=int, default=128,
                    help="patch size used for PromptIR tiled forward/gradient during adversarial generation")
parser.add_argument("--adv_promptir_patch_overlap", type=int, default=32,
                    help="patch overlap used for PromptIR tiled forward/gradient during adversarial generation")
parser.add_argument("--adv_cache_root", type=str,
                    default="/home/huhao/adv_ir/PromptIR/data/Train/adv_pairs",
                    help="root directory for generated adversarial input/target pair folders")
parser.add_argument("--adv_attack_device", type=str, default="cuda", choices=["cpu", "cuda"],
                    help="device used to generate adversarial samples")
parser.add_argument("--adv_map_interp_mode", type=str, default="bicubic",
                    choices=["nearest", "bilinear", "bicubic", "area", "gaussian"],
                    help="interpolation mode for lowres->highres control map in adversarial degradation")
parser.add_argument("--adv_gaussian_radius", type=int, default=4,
                    help="gaussian interpolation radius (effective when adv_map_interp_mode=gaussian)")
parser.add_argument("--adv_gaussian_sigma", type=float, default=1.25,
                    help="gaussian interpolation sigma (effective when adv_map_interp_mode=gaussian)")
parser.add_argument("--adv_gaussian_extra_cells", type=int, default=2,
                    help="extra boundary cells for gaussian interpolation (effective when adv_map_interp_mode=gaussian)")

options = parser.parse_args()

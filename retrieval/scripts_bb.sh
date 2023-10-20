
declare -a Datasets=( "iNat" "roxford5k" "INSTRE" "rparis6k" "google_landmark" "Copydays" "CUB200" )

for dset in ${Datasets[@]}; do


declare -a Models=(  "resnet50" "resnet101" "convnext_small" "convnext_base" "convnext_small_in22ft1k" "convnext_base_in22ft1k" "convnext_xlarge_in22ft1k" "vit_small_patch16_224_augreg_in1k" "vit_small_patch16_224_augreg_in21k_ft_in1k" "vit_base_patch16_224_augreg_in1k" "vit_base_patch16_224_augreg_in21k_ft_in1k" "vit_large_patch16_224_augreg_in21k_ft_in1k" "swinv2_tiny_window8_256" "swinv2_large_window12to16_192to256_22kft1k" "swinv2_base_window12to16_192to256_22kft1k" "swinv2_large_window12to24_192to384_22kft1k" "swinv2_base_window12to24_192to384_22kft1k" "vit_small_patch16_224_dino" "vit_base_patch16_224_dino" ) 

for model in ${Models[@]}; do



python main_ret.py --arch $model  --pt_style timm --dataset $dset --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64


done


python main_ret.py --arch vit_base_patch16_224  --pt_style clip --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 

python main_ret.py --arch vit_large_patch14_224  --pt_style clip --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 

python main_ret.py --arch resnet50  --pt_style clip --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 

python main_ret.py --arch resnet101  --pt_style clip --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 

python main_ret.py --arch resnet50_64  --pt_style clip --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 


python main_ret.py --arch vit_small_patch16_224  --pt_style moco --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 


python main_ret.py --arch vit_base_patch16_224  --pt_style moco --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 


python main_ret.py --arch vit_base_patch16_224  --pt_style mae --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 

python main_ret.py --arch vit_large_patch16_224  --pt_style mae --dataset $dset  --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64 

python main_ret.py --arch resnet50  --pt_style vicreg --dataset $dset --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64

python main_ret.py --arch swin2_tiny_256  --pt_style midas --dataset $dset --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64

python main_ret.py --arch swin2_base_384  --pt_style midas --dataset $dset --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64

python main_ret.py --arch resnet50_im  --pt_style sscd --dataset $dset --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64

python main_ret.py --arch resnet50_disc  --pt_style sscd --dataset $dset --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 # -b 64

done




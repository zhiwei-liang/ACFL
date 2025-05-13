# nohup python -u algorithms/fedavg/train_pacs.py --dataset pacs --num_classes 7 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >fedavg_pacs.log &
# nohup python -u algorithms/fedprox/train_pacs.py --dataset pacs --num_classes 7 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >fedprox_pacs.log &
# nohup python -u algorithms/scaffold/train_pacs.py --dataset pacs --num_classes 7 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >fedprox_pacs.log &
# nohup python -u algorithms/RSC/train_pacs.py --dataset pacs --num_classes 7 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18_rsc --note debug >fedrsc_pacs.log &
# nohup python -u   algorithms/ACFL/train_pacs_ACFL.py --dataset pacs --num_classes 7 &
# numclass==7

# nohup python -u algorithms/fedavg/train_pacs.py --dataset officehome --num_classes 65 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >fedavg_officehome.log &
# nohup python -u algorithms/fedprox/train_pacs.py --dataset officehome --num_classes 65 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >fedprox_officehome.log &
# nohup python -u algorithms/scaffold/train_pacs.py --dataset officehome --num_classes 65 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >scaffold_officehome.log &
# nohup python -u algorithms/RSC/train_pacs.py --dataset officehome --num_classes 65 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18_rsc --note debug >rsc_officehome.log &
# nohup python -u algorithms/ACFL/train_pacs_ACFL.py --dataset officehome --num_classes 65 --test_domain p --lr 0.001 --batch_size 32 --comm 200 --model resnet18 --note debug >acfl_officehome.log &

# nohup python -u algorithms/fedavg/train_pacs.py --dataset terrainc --num_classes 10 --test_domain 100 --lr 0.001 --batch_size 32 --comm 200 --model resnet50 --note debug >fedavg_terrainc.log &
# nohup python -u algorithms/fedprox/train_pacs.py --dataset terrainc --num_classes 10 --test_domain 100 --lr 0.001 --batch_size 32 --comm 200 --model resnet50 --note debug >fedprox_terrainc.log &
# nohup python -u algorithms/scaffold/train_pacs.py --dataset terrainc --num_classes 10 --test_domain 100 --lr 0.001 --batch_size 32 --comm 200 --model resnet50 --note debug >scaffold_terrainc.log &
# nohup python -u algorithms/RSC/train_pacs.py --dataset terrainc --num_classes 10 --test_domain 100 --lr 0.001 --batch_size 32 --comm 200 --model resnet50_rsc --note debug >rsc_terrainc.log &
nohup python -u algorithms/ACFL/train_pacs_ACFL.py --dataset terrainc --num_classes 10 --test_domain 100 --lr 0.001 --batch_size 32 --comm 200 --model resnet50 --note debug >acfl_terrainc.log &

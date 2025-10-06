#change src_step
python prompt_stable.py --gpu 0 --data_inds 4 --src_step=0 \
                        --ref_step1 0 --ref_step2 10 --mask_weight 3.0 --cross_step 0.6 &

python prompt_stable.py --gpu 1 --data_inds 4 --src_step=10 \
                        --ref_step1 0 --ref_step2 10 --mask_weight 3.0 --cross_step 0.6 &

python prompt_stable.py --gpu 2 --data_inds 4 --src_step=20 \
                        --ref_step1 0 --ref_step2 10 --mask_weight 3.0 --cross_step 0.6 &

python prompt_stable.py --gpu 3 --data_inds 4 --src_step=30 \
                        --ref_step1 0 --ref_step2 10 --mask_weight 3.0 --cross_step 0.6 &

python prompt_stable.py --gpu 4 --data_inds 4 --src_step=40 \
                        --ref_step1 0 --ref_step2 10 --mask_weight 3.0 --cross_step 0.6 &

python prompt_stable.py --gpu 5 --data_inds 4 --src_step=50 \
                        --ref_step1 0 --ref_step2 10 --mask_weight 3.0 --cross_step 0.6 &

#change cross_step
python prompt_stable.py --gpu 0 --data_inds 0 --src_step=15 \
                        --ref_step1 20 --ref_step2 50 --mask_weight 3.0 --cross_step 0.6 &

python prompt_stable.py --gpu 1 --data_inds 0 --src_step=15 \
                        --ref_step1 20 --ref_step2 50 --mask_weight 3.0 --cross_step 0.2 &

python prompt_stable.py --gpu 2 --data_inds 0 --src_step=15 \
                        --ref_step1 20 --ref_step2 50 --mask_weight 3.0 --cross_step 0.3 &

python prompt_stable.py --gpu 3 --data_inds 0 --src_step=15 \
                        --ref_step1 20 --ref_step2 50 --mask_weight 3.0 --cross_step 0.4 &

python prompt_stable.py --gpu 7 --data_inds 0 --src_step=15 \
                        --ref_step1 20 --ref_step2 50 --mask_weight 3.0 --cross_step 0.5

#search mode
python prompt_stable_v2.py --gpu 4 --data_inds 4 --src_step=30 \
                        --ref_step1 20 --ref_step2 50 --mask_weight 3.0 --cross_step 0.6 --ref_q_step 40 &

python prompt_stable_v2.py --data_inds 25 --src_step 30 --ref_q_step 30  --gpu 2 &    
python prompt_stable_v2.py --data_inds 25 --src_step 30 --ref_q_step 35  --gpu 3 &    
python prompt_stable_v2.py --data_inds 25 --src_step 30 --ref_q_step 40  --gpu 4 &    

python prompt_stable_v2.py --data_inds 0 --src_step 35 --ref_q_step 30 --gpu 6 &  
python prompt_stable_v2.py --data_inds 0 --src_step 35 --ref_q_step 45 --gpu 7 &

python prompt_stable_v2.py --data_inds 16  --src_step 50 --ref_q_step 35 --gpu 0  &
python prompt_stable_v2.py --data_inds 16  --src_step 50 --ref_q_step 40 --gpu 1  &
python prompt_stable_v2.py --data_inds 16  --src_step 45 --ref_q_step 35 --gpu 2  &
python prompt_stable_v2.py --data_inds 16  --src_step 45 --ref_q_step 40 --gpu 3  &

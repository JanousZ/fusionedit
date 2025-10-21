#new version
export CUDA_VISIBLE_DEVICES=1
python prompt_stable_v2.py --expid 10 \
    --self_ref_kv_time_end 50 --self_ref_kv_layer_idx 0 16 \
    --self_ref_q_time_end 40 --self_ref_q_layer_idx 0 16 \
    --self_src_q_time_end 45 --self_src_q_layer_idx 0 16 \
    --mask_weight 3.0 \
    --cross_step 0.6 --cross_src_layer_idx 0 16

暂时来说 cross部分先不动，因为这是前人验证过的，并且代码对应逻辑还未完善

寻找：对应的所有layer信息 self，cross以及大小，以及对应的id。
0 self torch.Size([6, 4096, 320])   64
1 cross torch.Size([6, 4096, 320])
2 self torch.Size([6, 4096, 320])
3 cross torch.Size([6, 4096, 320])
4 self torch.Size([6, 1024, 640])   32
5 cross torch.Size([6, 1024, 640])
6 self torch.Size([6, 1024, 640])
7 cross torch.Size([6, 1024, 640])  
8 self torch.Size([6, 256, 1280])   16
9 cross torch.Size([6, 256, 1280])
10 self torch.Size([6, 256, 1280])
11 cross torch.Size([6, 256, 1280])  
12 self torch.Size([6, 64, 1280])   8
13 cross torch.Size([6, 64, 1280])
14 self torch.Size([6, 256, 1280])  16
15 cross torch.Size([6, 256, 1280])
16 self torch.Size([6, 256, 1280])
17 cross torch.Size([6, 256, 1280])
18 self torch.Size([6, 256, 1280])
19 cross torch.Size([6, 256, 1280])
20 self torch.Size([6, 1024, 640])  32
21 cross torch.Size([6, 1024, 640])
22 self torch.Size([6, 1024, 640])
23 cross torch.Size([6, 1024, 640])
24 self torch.Size([6, 1024, 640])
25 cross torch.Size([6, 1024, 640])
26 self torch.Size([6, 4096, 320])  64
27 cross torch.Size([6, 4096, 320])
28 self torch.Size([6, 4096, 320])
29 cross torch.Size([6, 4096, 320])
30 self torch.Size([6, 4096, 320])
31 cross torch.Size([6, 4096, 320])

注意 ref_q当时是除了中间层其他都替换了，跟现在的range逻辑不一样

结论1:expid1，self_src_q_time_end不能太大,至少不能是50，不然原图中的细节会被完整保留（有残留）。    40
结论2:expid1, self_ref_q_layer_idx不能太少，至少不能是[8,9),不然换不过来。
结论3:self_ref_q_time_end也不要太大,至少不能是50，因为它是不准确的，如果太大会造成最终的模糊。     40
而且self_ref_q和self_src_q之间应该有一个大小关系
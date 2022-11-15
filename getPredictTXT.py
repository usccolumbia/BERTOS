import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(
    description="Test trained model."
)
parser.add_argument(
    "--ref_path",
    type=str,
    default=None,
    help='./dataset/finetune/ICSD_oxide/test.txt',
)

parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="./finetune/ft_icsd/predict_icsdo/",
    required=False,
)

args = parser.parse_args()


f = open(args.ref_path)

mt = []
state = []

lines = f.readlines()
f.close()

for line in lines:
    #print(line
    if line == '\n':
        mt.append(' ')       
        state.append(' ')
    else:
        mt.append(line.strip().split()[0])       
        state.append(line.strip().split()[1])


# In[5]:


mt1 = []
tmp = []
for ele in mt:
    if ele != ' ':
        tmp.append(ele)
    else:
        mt1.append(tmp)
        tmp = []



from ast import literal_eval
df = pd.read_csv(os.path.join(args.save_path, 'results.csv'),header=None)   # (input_id, refs, pred, probs)

target_state = []
pred_state = []
prob_state = []
for i in range(len(df)):
    if i != len(df)-1:
        target_state.append(literal_eval(df.iloc[i][1]))
        pred_state.append(literal_eval(df.iloc[i][2]))
        prob_state.append(literal_eval(df.iloc[i][3]))

print(len(mt1), len(target_state))

count1 = 0
count2 = 0

mt_new = []
targets_new = []
preds_new = []
prob_new = []
for j in range(len(mt1)):
    if len(mt1[j]) == len(target_state[j]):
        count1 += 1
        mt_new.append(mt1[j])
        targets_new.append(target_state[j])
        preds_new.append(pred_state[j])
        prob_new.append(prob_state[j])
        #print(len(mt1[j]), len(target_state[j]), mt1[j], target_state[j])
    #if len(pred_state[j]) != len(target_state[j]):
    #    count2 += 1
print('Materials larger than 100: ', len(mt1)-count1)




import os
if os.path.exists(os.path.join(args.save_path, 'res.txt')):
    os.remove(os.path.join(args.save_path, 'res.txt'))

with open(os.path.join(args.save_path, 'res.txt'), "a") as f:
    for i in range(len(mt_new)):
        str_mt = ' '.join(mt_new[i])
        
        tmp_t = list(map(int, targets_new[i]))
        tmp_t = (np.array(tmp_t)).tolist()
        tmp_t1 = list(map(str, tmp_t))
        str_targ = ' '.join(tmp_t1)
        
        
        tmp_p = list(map(int, preds_new[i]))
        tmp_p = (np.array(tmp_p)).tolist()
        tmp_p1 = list(map(str, tmp_p))
        str_pred = ' '.join(tmp_p1)

        tmp_prob = list(map(float, prob_new[i]))
        tmp_prob = (np.array(tmp_prob)).tolist()
        tmp_prob1 = list(map(str, tmp_prob))
        str_prob = ' '.join(tmp_prob1)
    
        f.write(str_mt)
        f.write('\n')
        f.write(str_targ)
        f.write('\n')
        f.write(str_pred)
        f.write('\n')
        f.write(str_prob)
        f.write('\n')
        f.write('\n')
f.close()

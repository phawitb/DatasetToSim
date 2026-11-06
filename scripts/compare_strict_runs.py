"""Compare all-episode outcomes and frame-weighted native state errors."""
import argparse,json
from pathlib import Path
import numpy as np
from dataset_to_sim.identification import held_gripper_frames


def compare(baseline,candidate,output):
 folders=[Path(baseline),Path(candidate)];reports=[json.loads((p/'results.json').read_text())for p in folders];summaries=[]
 for folder,report in zip(folders,reports):
  frames=sum(r['frames']for r in report['results']);rmse=np.sqrt(sum(np.array(r['joint_state_rmse_native'])**2*r['frames']for r in report['results'])/frames);held=[]
  for r in report['results']:
   t=np.load(folder/f'episode-{r["episode"]:02d}/strict_trace.npz');state=t['recorded_native_state'];mask=held_gripper_frames(state[:,5],t['native_action_applied'][:,5]);held.extend((t['simulated_native_state'][:-1,5][mask]-state[mask,5]).tolist())
  summaries.append({'folder':str(folder),'successes':report['successes'],'counts':report['counts'],'frames':frames,'joint_rmse_native':rmse.tolist(),'arm_pooled_rmse_deg':float(np.sqrt(np.mean(rmse[:5]**2))),'held_gripper_rmse_native':float(np.sqrt(np.mean(np.array(held)**2))),'initial_max_penetration_m':max(r['audit']['initial_max_penetration_m']for r in report['results']),'failed_episodes':[r['episode']for r in report['results']if r['outcome']!='success']})
 before={r['episode']:r['outcome']for r in reports[0]['results']};after={r['episode']:r['outcome']for r in reports[1]['results']};assert before.keys()==after.keys()
 result={'baseline':summaries[0],'candidate':summaries[1],'recovered':[i for i in before if before[i]!='success'and after[i]=='success'],'regressed':[i for i in before if before[i]=='success'and after[i]!='success'],'episode_outcomes':[{'episode':i,'before':before[i],'after':after[i]}for i in before],'scope':'same calibration dataset, not blind evaluation or validated physical transfer'}
 Path(output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='episode_outcomes'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--output',required=True);a=p.parse_args();compare(a.baseline,a.candidate,a.output)

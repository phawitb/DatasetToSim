"""Read-only inventory of every saved Studio capture; never activates a run."""
import json,math
from pathlib import Path


def list_runs(work,jobs):
    rows=[]
    for folder in Path(work).iterdir():
        identifier=folder.name
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier) or not (folder/'capture.json').is_file():continue
        errors=[]
        def read(name):
            path=folder/name
            if not path.exists():return {}
            try:
                value=json.loads(path.read_text())
                if not isinstance(value,dict):raise ValueError('Expected report object')
                return value
            except (OSError,ValueError):errors.append(f'Could not read {name}');return {}
        capture=read('capture.json');plan=read('plan.json');execution=read('execution.json');physical=read('physical-review.json');job=read('job.json')|jobs.get(identifier,{})
        if not execution:execution=plan.get('execution',{})
        trace=read('hardware-trace.json')
        command_count=len(trace.get('samples',[]))
        if not command_count and (folder/'hardware-commands.jsonl').exists():
            for line in (folder/'hardware-commands.jsonl').read_text().splitlines():
                try:
                    sample=json.loads(line)
                    if isinstance(sample.get('target'),list) and len(sample['target'])==6:command_count+=1
                except ValueError:continue
        if capture.get('source')!='live' or command_count==0:continue
        if not trace and not execution and job.get('status') not in ['executing','recording']:trace={'completed':False,'error':'Execution interrupted by server restart; recording unavailable'}
        completed=execution.get('completed',trace.get('completed'))
        physical_outcome=physical.get('assessment',physical.get('task_outcome'))
        if physical_outcome in ['not_transferred','failure']:physical_outcome='failed'
        status=job.get('status') or ('complete' if completed is True else 'stopped' if completed is False else 'simulated' if plan.get('outcome') not in [None,'not_simulated'] else 'planned' if plan else 'incomplete' if capture.get('purpose')=='home' else 'captured')
        video=execution.get('video') or plan.get('video')
        if not isinstance(video,str) or Path(video).name!=video or not (folder/video).is_file():video=None
        stamp=capture.get('captured_at',folder.stat().st_mtime)
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):stamp=folder.stat().st_mtime
        rows.append({'id':identifier,'captured_at':stamp,'source':capture.get('source','unknown'),'purpose':capture.get('purpose','pick_place'),'status':'error' if errors else status,'stage':job.get('stage'),'simulation_outcome':plan.get('outcome'),'hardware_completed':completed,'physical_outcome':physical_outcome,'frames':plan.get('frames'),'duration':plan.get('duration'),'video':f'/media/studio/{identifier}/{video}' if video else None,'preview':f'/media/studio/{identifier}/top.jpg' if (folder/'top.jpg').exists() else None,'error':'; '.join(errors) or job.get('error') or execution.get('error') or trace.get('error')})
        rows[-1]['command_count']=command_count
        rows[-1]['speed_scale']=plan.get('speed_scale',1.)
        rows[-1]['task']=plan.get('task',capture.get('task','bowl'))
        rows[-1]['motion_technique']=plan.get('motion_technique','classic') if capture.get('purpose')!='home' else None
    return sorted(rows,key=lambda r:(r['captured_at'],r['id']),reverse=True)

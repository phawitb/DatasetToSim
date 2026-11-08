"""Accepted-example quotas and spatial stratification, independent of physics gates."""
from collections import Counter
import numpy as np


def task_requests(options, episodes):
    existing=Counter(item['task_id'] for item in episodes if item.get('status')=='accepted')
    targets=options.get('target_counts')
    if targets is None:return {task:options['episodes_per_task'] for task in options['tasks']}
    if any(existing[task]>targets[task] for task in options['tasks']):
        raise ValueError('Requested total is smaller than already accepted episodes')
    return {task:targets[task]-existing[task] for task in options['tasks']}


class BalancedLayouts:
    """Balance source/destination marginals across all nine full-mat cells.

    Failed pairs receive a retry cost, never a success count. No cell is removed
    for being difficult; exhausting the worker budget exposes a real deficit.
    """
    def __init__(self, world, seed, target, source_cells=None, target_cells=None):
        mat=world['mat'];self.low=np.asarray(mat['position'][:2])-np.asarray(mat['size'][:2])/2
        self.size=np.asarray(mat['size'][:2]);self.rng=np.random.default_rng(seed)
        self.source=np.zeros(9,dtype=int);self.target=np.zeros(9,dtype=int)
        self.failures=np.zeros((9,9),dtype=int);self.total=target
        self.source_cells=list(range(9)) if source_cells is None else list(source_cells)
        self.target_cells=list(range(9)) if target_cells is None else list(target_cells)
        if any(not cells or any(type(c) is not int or not 0<=c<9 for c in cells) or len(set(cells))!=len(cells)
               for cells in (self.source_cells,self.target_cells)):
            raise ValueError('Balanced zones must be distinct cell IDs from 0 to 8')
        self.quotas=np.zeros(9,dtype=int);self.target_quotas=np.zeros(9,dtype=int)
        self.prepare_additions(target)
    def prepare_additions(self,count):
        """Allocate new accepts only; inherited positions remain immutable.

        Water filling accounts for already imbalanced or newly excluded cells
        without spending the new batch's quota on unreachable/excluded cells.
        """
        def allocate(existing,cells):
            quotas=existing.copy()
            for _ in range(count):
                candidates=np.asarray(cells);values=quotas[candidates]
                selected=int(self.rng.choice(candidates[values==values.min()]))
                quotas[selected]+=1
            return quotas
        self.quotas=allocate(self.source,self.source_cells)
        self.target_quotas=allocate(self.target,self.target_cells)
    def cell(self,xy):
        ij=np.clip(np.floor((np.asarray(xy)-self.low)/self.size*3).astype(int),0,2)
        return int(ij[1]*3+ij[0])
    def accept_world(self,world):self.accept_cells(self.cell(world['cube']['position'][:2]),self.cell(world['target_cube']['position'][:2]))
    def accept_cells(self,source,target):self.source[source]+=1;self.target[target]+=1
    def choose(self):
        source=np.flatnonzero(self.source<self.quotas);target=np.flatnonzero(self.target<self.target_quotas)
        if not len(source) or not len(target):raise ValueError('Balanced layout quota is already full')
        choices=[(int(a),int(b)) for a in source for b in target]
        source_left=np.maximum(0,self.quotas-self.source);target_left=np.maximum(0,self.target_quotas-self.target)
        remaining=int(source_left.sum())
        # Preserve a feasible off-diagonal matching, so filling easier zones
        # cannot leave a final pick/place pair trapped in the same small zone.
        if remaining == int(target_left.sum()) and np.max(source_left+target_left) <= remaining:
            feasible=[]
            for a,b in choices:
                if a == b:continue
                totals=source_left+target_left;totals=totals.copy();totals[a]-=1;totals[b]-=1
                if np.max(totals) <= remaining-1:feasible.append((a,b))
            if feasible:choices=feasible
        costs=np.array([self.source[a]+self.target[b]+.7*np.sqrt(self.failures[a,b])+(a==b)*.5 for a,b in choices])
        best=np.flatnonzero(np.isclose(costs,costs.min()));a,b=choices[int(self.rng.choice(best))]
        return {'grid_shape':[3,3],'source_cell':a,'target_cell':b}
    def reject(self,choice):self.failures[choice['source_cell'],choice['target_cell']]+=1
    def summary(self):
        return {'grid_shape':[3,3],'source_counts':self.source.tolist(),'target_counts':self.target.tolist(),
                'per_cell_targets':self.quotas.tolist(),'source_cell_targets':self.quotas.tolist(),
                'target_cell_targets':self.target_quotas.tolist(),'allowed_source_cells':self.source_cells,
                'allowed_target_cells':self.target_cells,'pair_rejections':self.failures.tolist()}

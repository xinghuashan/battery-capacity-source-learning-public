"""Atomic source pseudo-groups for the M06 control."""
import hashlib
from functools import lru_cache
import numpy as np

def pseudo_assign(source):
    """Lexicographically first exact atomic allocation in fixed SHA order."""
    ids, refs, selected = source['ids'], source['reference_raw_record'], source['chosen']
    groups = sorted(np.unique(ids).tolist())
    atoms = {}
    for i,(cell,ref) in enumerate(zip(ids,refs)):
        atoms.setdefault((int(cell),int(ref)),[]).append(i)
    ordered = sorted(atoms,key=lambda key:(hashlib.sha256(f'battery-meta-placebo-v1|cell={key[0]}|reference={key[1]}'.encode()).hexdigest(),key))
    capacities_u = tuple(int((ids==cell).sum()) for cell in groups)
    capacities_l = tuple(int(((ids==cell)&selected).sum()) for cell in groups)
    blocks = [atoms[key] for key in ordered]
    @lru_cache(None)
    def visit(position,u,l,physical_bits,mixed):
        if position==len(blocks): return () if mixed and not any(u) and not any(l) else None
        rows=blocks[position];size=len(rows);label=int(selected[rows].sum());physical=groups.index(int(ids[rows[0]]))
        for group in range(len(groups)):
            if u[group]<size or l[group]<label or u[group]-l[group]<size-label:continue
            nu,nl=list(u),list(l);nu[group]-=size;nl[group]-=label
            bits=list(physical_bits);bits[physical]|=1<<group
            now=mixed or bits[physical].bit_count()>1
            tail=visit(position+1,tuple(nu),tuple(nl),tuple(bits),now)
            if tail is not None:return (group,)+tail
        return None
    assignment=visit(0,capacities_u,capacities_l,(0,)*len(groups),False)
    if assignment is None:return dict(available=False,reason='No exact atomic allocation with real-cell cross-split')
    labels=np.empty(len(ids),dtype=int)
    records=[]
    for key,rows,group in zip(ordered,blocks,assignment):
        labels[rows]=group
        records.append(dict(physical_cell=key[0],reference_raw_record=key[1],rows=rows,selected_labels=int(selected[rows].sum()),
                            hash=hashlib.sha256(f'battery-meta-placebo-v1|cell={key[0]}|reference={key[1]}'.encode()).hexdigest(),group=group))
    details=[]
    for group,cell in enumerate(groups):
        chosen=labels==group
        assert int(chosen.sum())==capacities_u[group] and int((chosen&selected).sum())==capacities_l[group]
        details.append(dict(group=group,matched_real_cell=cell,U=int(chosen.sum()),L=int((chosen&selected).sum()),
                            physical_composition={str(c):dict(U=int((chosen&(ids==c)).sum()),L=int((chosen&(ids==c)&selected).sum())) for c in groups}))
    split=[int(cell) for cell in groups if len(np.unique(labels[ids==cell]))>1]
    assert split
    return dict(available=True,group_order=groups,labels=labels.tolist(),ordered_blocks=records,groups=details,
                cross_split_real_cells=split,algorithm='ordered atomic blocks, ascending group DFS, first exact cross-split feasible assignment',
                hash_encoding='UTF8 battery-meta-placebo-v1|cell={physical}|reference={actual_reference_raw_record}')

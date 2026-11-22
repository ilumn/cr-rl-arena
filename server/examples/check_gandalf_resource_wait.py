"""Fast deterministic resource-hold, physical discounting and PPO checks."""

import contextlib
import copy
import io
from unittest.mock import patch

import cr_sim
import torch

import gandalf_resource_wait as resource
import train_gandalf as core
from gandalf import CARD_NAMES, DECK, action_masks, encode_values
from gandalf_coordinate import CoordinateActorCritic
from model_brains import select_action


def observation(tick=0, elixir=3., threat=False, hp=4000, legal=None):
    entities = [[0,'king_tower',9.,3.,hp,4000,False,True,False,1,2.,2.]]
    if threat:
        entities.append([1,'knight',3.,16.,100,100,False,False,False,2,0.,0.])
    return dict(tick=tick,elixir=elixir,hand=['knight','musketeer','giant','archers'],
                hand_cost=[3,4,5,3],legal=legal or [True,False,False,True],entities=entities)


def check_controller():
    hold=resource.ResourceHold(60)
    assert hold.start(observation()) and hold.target==4.
    assert hold.continue_holding(observation(20,3.5))
    assert not hold.continue_holding(observation(40,4.,legal=[True,True,False,True]))
    assert hold.counts['released_card_unlocked']==1
    for kind in ['incoming_threat','tower_damage','time_limit']:
        hold=resource.ResourceHold(2)
        assert hold.start(observation())
        obs=observation(2 if kind=='time_limit' else 1,3.1,
                        threat=kind=='incoming_threat',hp=3990 if kind=='tower_damage' else 4000)
        assert not hold.continue_holding(obs)
        assert hold.counts[f'released_{kind}']==1
    hold=resource.ResourceHold(60)
    assert not hold.start(observation(threat=True)) and not hold.active
    assert not hold.start(observation(elixir=10.,legal=[True]*4))
    assert hold.start(observation(elixir=5.,legal=[True]*4)) and hold.target==6.
    assert not hold.continue_holding(observation(40,6.,legal=[True]*4))
    print('PASS hold: new-card release, time cap, incoming threats, tower damage, existing danger and elixir cap')


class FakeBattle:
    threat_at=None
    tower_damage_at=None
    finish_at=6

    def reset(self,**kwargs):
        self.tick=0
        self.spent=False

    def view(self,seat):
        return self

    def observe(self):
        legal=[False]*4 if self.spent else [True,self.tick>=3,False,True]
        obs=observation(self.tick,3.+self.tick/3,
                        threat=self.threat_at is not None and self.tick>=self.threat_at,
                        hp=3990 if self.tower_damage_at is not None and self.tick>=self.tower_damage_at else 4000)
        obs['legal']=legal
        obs['last_action']='none'
        return obs

    def step(self,a=None,b=None):
        self.spent |= a is not None or b is not None
        self.tick+=1

    def done(self):
        return self.tick>=self.finish_at

    def result(self):
        return 'win' if self.done() else 'running'


def fake_rollout(mode,threat=None,damage=None,finish=6,limit=60):
    selected=[]
    def sampler(model,encoded,slots,tiles):
        selected.append(encoded['tick'])
        return (0,None,-.5,0.) if len(selected)==1 else (1,0,-.5,0.)
    def masks(obs):
        return [True,*obs['legal']],[[bool(legal)] for legal in obs['legal']]
    with patch.object(FakeBattle,'threat_at',threat), patch.object(FakeBattle,'tower_damage_at',damage), \
            patch.object(FakeBattle,'finish_at',finish), patch.object(resource.cr_sim,'Battle',FakeBattle), \
            patch.object(resource,'select_action',sampler),patch.object(resource,'action_masks',masks), \
            patch.object(resource,'encode_values',lambda obs:dict(tick=obs['tick'])):
        records,result,stats=resource.collect_resource_match(
            None,1,lambda obs:None,DECK,wait_mode=mode,max_hold_ticks=limit,gamma=.9)
    return records,result,stats,selected


def check_timing():
    records,result,stats,selected=fake_rollout('reserve')
    assert selected==[0,3] and [r['duration_ticks'] for r in records]==[3,3]
    assert result=='win' and stats['counts']['physical_ticks']==6
    assert stats['counts']['forced_hold_ticks']==2
    assert stats['counts']['forced_unaffordable_ticks']==2
    assert len(records)==2 and all(sum(r['slot_mask'])>1 for r in records)
    tick_records=[dict(reward=1. if i==5 else 0.,done=i==5,value_estimate=0.) for i in range(6)]
    full,_=core.calculate_returns_and_advantages(tick_records,gamma=.9,gae_lambda=1.)
    returns,_=resource.decision_returns(records,.9)
    assert torch.allclose(returns,full[torch.tensor(selected)],atol=1e-6)
    control,_,_,selected=fake_rollout('tick')
    assert selected==[0,1] and [r['duration_ticks'] for r in control]==[1,5]
    control_returns,_=resource.decision_returns(control,.9)
    assert torch.allclose(control_returns,full[torch.tensor(selected)],atol=1e-6)
    for kwargs,reason in [({'threat':2},'incoming_threat'),({'damage':2},'tower_damage'),({'limit':2},'time_limit')]:
        actual,_,info,calls=fake_rollout('reserve',**kwargs)
        assert calls==[0,2] and actual[0]['duration_ticks']==2
        assert info['holds'][f'released_{reason}']==1
    early,_,info,calls=fake_rollout('reserve',finish=2)
    assert calls==[0] and early[0]['done'] and early[0]['duration_ticks']==2
    assert info['holds']['released_match_end']==1
    assert torch.allclose(resource.decision_returns(early,.9)[0],torch.tensor([.9]))
    episodes=records+control
    together,_=resource.decision_returns(episodes,.9)
    assert torch.allclose(together,torch.cat([returns,control_returns]))
    print('PASS timing: forced ticks excluded; immediate interruption; exact full-tick MC equivalence; terminal holds and episode boundaries')


def check_ppo():
    torch.manual_seed(123)
    battle=cr_sim.Battle(); battle.reset(seed=42,deck0=DECK,deck1=DECK)
    obs=battle.view(0).observe()
    encoded=encode_values(obs); slots,tiles=action_masks(obs)
    model=CoordinateActorCritic(len(CARD_NAMES))
    records=[]
    for i in range(8):
        slot,tile,log,value=select_action(model,encoded,slots,tiles)
        duration=3 if slot==0 else 1
        records.append(dict(observations=encoded,slot_action=slot,tile_id=tile,
                            slot_mask=slots,tile_masks=tiles,log_probability=log,
                            value_estimate=value,duration_ticks=duration,
                            reward=.9999**(duration-1)*(1. if i%2 else -1.),done=True))
    returns,advantages=resource.decision_returns(records,.9999)
    initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=core.SETTINGS['learning_rate'])
    with patch.dict(core.SETTINGS,dict(ppo_epochs=1,actor_batch_size=8,critic_batch_size=8)), \
            contextlib.redirect_stdout(io.StringIO()):
        core.ppo_update(model,optimizer,records,returns,advantages)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters())
    print('PASS PPO: unchanged sampler/recomputed joint probabilities; finite decision-state actor/critic updates')


def main():
    torch.set_num_threads(1)
    check_controller()
    check_timing()
    check_ppo()
    print('All resource-wait checks passed. No model checkpoints modified.')


if __name__=='__main__':
    main()

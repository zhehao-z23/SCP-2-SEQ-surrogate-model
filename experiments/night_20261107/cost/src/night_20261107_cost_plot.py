"""Anonymous aggregate cost plots; no individual inputs or model outputs."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
j=json.loads(Path(a.summary).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
s=j['length_frame']['strata'];labels=['40–150','151–300','301–600','601–1200','1201–2400','2401–3619']
x=np.arange(6);enc=np.array([r['encoder_seconds_mean_warm'] for r in s])*1000
read=np.array([r['readout_seconds_mean_warm'] for r in s])*1000
total=np.array([np.mean(r['pipeline_seconds_per_sequence_by_repeat'][1:]) for r in s])*1000
fig,ax=plt.subplots(figsize=(9,5));ax.bar(x,enc,label='Encoder (includes CPU output)');ax.bar(x,read,bottom=enc,label='Readout + normalization + transfers');ax.bar(x,total-enc-read,bottom=enc+read,label='Descriptors / conversion / checks')
for i,r in enumerate(s):ax.text(i,total[i]+2,f"n={r['sample_n']}",ha='center',fontsize=9)
ax.set_xticks(x,labels);ax.set_xlabel('Input length stratum (amino acids)');ax.set_ylabel('Measured warm milliseconds per sequence');ax.set_ylim(0,max(total)*1.14);ax.legend(frameon=False,loc='upper left');ax.set_title('95 full sequences: frozen ESMC300 + residue readout, one RTX 4090')
fig.text(.01,.01,'Two warm passes; FP32 singleton encoder, FP16 residue roundtrip. Historical Atlas input frame; no accuracy or AFDB-wide claim.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'length_cost.png',dpi=180);fig.savefig(out/'length_cost.svg');plt.close(fig)
d=j['conditional_scan_scenarios'][1]['storage_payload_bytes'];names=['63 probabilities\nFP32','Sequence letters\n(no headers)','Mean vectors\nFP32','Full residues\nFP16'];vals=[d['probabilities63_FP32'],d['sequence_residue_letters_only'],d['mean960_FP32'],d['residue960_FP16']]
fig,ax=plt.subplots(figsize=(8,5));ax.bar(names,np.array(vals)/1e9,color=['#14857c','#617d8a','#d88b32','#b34f50']);ax.set_yscale('log');ax.set_ylabel('Payload storage (decimal GB, log scale)');ax.set_title('Conditional payload at 100 million sequences')
for i,y in enumerate(vals):ax.text(i,y/1e9*1.16,f'{y/1e9:,.1f} GB',ha='center',fontsize=10)
ax.set_ylim(10,200000);fig.text(.01,.01,'Assumes the observed 290.15-aa mean. Excludes metadata, replicas and filesystem overhead; residue caching is optional.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'scan_storage.png',dpi=180);fig.savefig(out/'scan_storage.svg');plt.close(fig)
print(json.dumps({'status':'complete','files':['length_cost.png','length_cost.svg','scan_storage.png','scan_storage.svg'],'new_scientific_computation':False}))

"""Small fixed-scale diagnostics for stationary optical search runs."""
import json
import math


def save_diagnostics(out, space, cycle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=out/"visualizations"
    target.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(space.layers),4,figsize=(15,3.5*len(space.layers)),squeeze=False)
    for i,layer in enumerate(space.layers):
        base=space.base[i].detach().squeeze().cpu()
        current=layer.phase_raw.detach().squeeze().cpu()
        difference=current-base
        circular=difference.sin().atan2(difference.cos())
        rms=circular.square().mean().sqrt().item()
        peak=circular.abs().max().item()
        zoom_range = max(3 * rms, 0.005)
        arrays=((base%(2*math.pi),"Initial phase",0,2*math.pi,"viridis"),
                (current%(2*math.pi),"Current phase",0,2*math.pi,"viridis"),
                (circular,"Circular difference",-math.pi,math.pi,"RdBu_r"),
                (circular,f"Zoom (clipped to +/-{zoom_range:.4f} rad)\nRMS={rms:.5f}, max={peak:.5f} rad",-zoom_range,zoom_range,"RdBu_r"))
        for j,(data,title,low,high,cmap) in enumerate(arrays):
            plot=axes[i,j].imshow(data.numpy(),vmin=low,vmax=high,cmap=cmap)
            axes[i,j].set_title(f"SLM {i+1}: {title}")
            axes[i,j].axis("off")
            fig.colorbar(plot,ax=axes[i,j],shrink=.7,label="rad")
    fig.tight_layout()
    fig.savefig(target/f"phase_cycle_{cycle:03d}.png",dpi=150)
    plt.close(fig)
    rows=[json.loads(x) for x in (out/"cycles.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    initial=json.loads((out/"initial_metrics.json").read_text(encoding="utf-8"))["map50"]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.axhline(initial,color="gray",linestyle="--",label="Initialization")
    ax.plot([x["cycle"] for x in rows],[x["metrics"]["map50"] for x in rows],"o-",label="End-of-cycle candidate")
    ax.plot([x["cycle"] for x in rows],[x["best_map50"] for x in rows],label="Best retained pair")
    ax.set(xlabel="Cycle",ylabel="Validation mAP50")
    ax.legend();fig.tight_layout();fig.savefig(target/"map50.png",dpi=150);plt.close(fig)
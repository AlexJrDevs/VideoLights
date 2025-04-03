import torch
from video_lights.model import build_transformer, build_position_encoding, VideoLight

def build_inference_model(ckpt_path, **kwargs):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["opt"]


    if "query_embed.weight" in ckpt["model"]:
        query_embed_shape = ckpt["model"]["query_embed.weight"].shape
        args.num_queries = query_embed_shape[0]  # Number of queries from checkpoint

        print(f"Query embed shape: {query_embed_shape[0]}, num_queries: {args.num_queries}")

    if len(kwargs) > 0:  # Used to overwrite default args
        args.update(kwargs)

    # Build transformer and position encoding
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)



    model = VideoLight(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,

        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )

    # Load state_dict with strict=False to allow for size mismatches in keys
    model.load_state_dict(ckpt["model"], strict=False)
    return model

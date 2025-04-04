from huggingface_hub import snapshot_download
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions
from collections import defaultdict
import os
import glob
import shutil
import random

class FeatureRetriever:
    def __init__(self,
                 num_images=50,
                 imgs_per_dir=15,
                 force_download=False):

        if force_download or not os.path.exists("./clip"):
            print("Downloading clip resources")
            rand_num = random.randint(0, 100000)
            tmp_dir = f"./tmp_{rand_num}"
            snapshot_download(repo_type="dataset", repo_id="wendlerc/sdxl-unbox-clip-indices", cache_dir=tmp_dir)
            clip_dirs = glob.glob(f"{tmp_dir}/**/down_10_5120", recursive=True)
            if len(clip_dirs) > 0:
                shutil.copytree(clip_dirs[0].replace("down_10_5120", ""), "./clip", dirs_exist_ok=True)
                shutil.rmtree(tmp_dir)
            else:
                ValueError("Could not find clip indices in the downloaded repo.")

        # Initialize CLIP service
        clip_options = ClipOptions(
            indice_folder="currently unused by knn.query()",
            clip_model="ViT-B/32", #"open_clip:ViT-H-14",
            enable_hdf5=False,
            enable_faiss_memory_mapping=True,
            columns_to_return=["image_path", "similarity"],
            reorder_metadata_by_ivf_index=False,
            enable_mclip_option=False,
            use_jit=False,
            use_arrow=False,
            provide_safety_model=False,
            provide_violence_detector=False,
            provide_aesthetic_embeddings=False,
        )
        self.names = ["down.2.1", "mid.0", "up.0.0", "up.0.1"]
        self.paths = ["./clip/down_10_5120/indices_paths.json",
                 "./clip/mid_10_5120/indices_paths.json",
                 "./clip/up0_10_5120/indices_paths.json",
                 "./clip/up_10_5120/indices_paths.json",]
        self.knn_service = {}
        for name, path in zip(self.names, self.paths):
            resources = load_clip_indices(path, clip_options)
            self.knn_service[name] = KnnService(clip_resources=resources)
        self.num_images = num_images
        self.imgs_per_dir = imgs_per_dir

    def query_text(self, query, block):
        if block not in self.names:
            raise ValueError(f"Block must be one of {self.names}")
        results = self.knn_service[block].query(
            text_input=query,
            num_images=self.num_images,
            num_result_ids=self.num_images,
            deduplicate=True,
        )
        feat_sims = defaultdict(list)
        feat_scores = {}
        for result in results:
            feature_id = result["image_path"].split("/")[-2]
            feat_sims[feature_id] += [result["similarity"]]
        for fid, sims in feat_sims.items():
            feat_scores[fid] = (sum(sims) / len(sims)) * (len(sims)/self.imgs_per_dir)

        return dict(sorted(feat_scores.items(), key=lambda item: -item[1]))

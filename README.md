# RecDCL: Dual Contrastive Learning for Recommendation

<p align="center">
📃 <a href="https://arxiv.org/abs/2401.15635" target="_blank">[RecDCL]</a> <a href="https://github.com/THUDM/RecDCL" target="_blank">[GitHub]</a> <br>
</p>

**RecDCL** is a dual CL method for recommendation, which investigates the relationship of objectives between batch-wise CL (BCL) and feature-wise CL (FCL). Our study suggests a cooperative benefit of employing both methods, as evidenced from theoretical and experimental perspectives. RecDCL first eliminates redundant solutions on user-item positive pairs in a feature-wise manner. It then optimizes the uniform distributions within users and items using a polynomial kernel from an FCL perspective. Finally, it generates contrastive embedding on output vectors in a batch-wise objective. RecDCL achieves more competitive performance than the state-of-the-art GNNs-based and SSL-based models (with up to a 5.65\% improvement in terms of Recall@20).

![](./assets/BCL_FCL.png)

![](./assets/framework.png)

## **Table of Contents**

- [PyTorch Implementation](#Implementation)
- [Leaderboard](#Leaderboard)
- [Citation](#Citation)

## **PyTorch Implementation**

This is our PyTorch implementation for the WWW'24 paper:

### Environment Requirements

The code has been tested running under Python 3.10.9. The required packages are as follows:

- PyTorch == 2.0.1

### Usage Example
#### Running one trial on Beauty:

```
bash run_beauty.sh
```
## **Leaderboard**

Overall top-20 performance comparison with representative models on four datasets.

![](./assets/RecDCL_results.png)


## Citation
If you find our work helpful, please kindly cite our paper:

```
@misc{zhang2024recdcl,
      title={RecDCL: Dual Contrastive Learning for Recommendation}, 
      author={Dan Zhang and Yangliao Geng and Wenwen Gong and Zhongang Qi and Zhiyu Chen and Xing Tang and Ying Shan and Yuxiao Dong and Jie Tang},
      year={2024},
      eprint={2401.15635},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

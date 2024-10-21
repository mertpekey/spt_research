# Sp Trs Draft Notes

Date: October 14, 2024
Status: Not started

<aside>
💡

**Key Concepts (Summarized by GPT4o):**

1. **Spatial Transcriptomics**:
    - It refers to the study of gene expression in the context of tissue structure. In other words, it’s about understanding which genes are active and where they are active within a tissue sample.
    - Imagine a tissue sample (like a slice of breast tissue). You divide this sample into small spots or regions.
    - For each spot, you measure the **gene expression** (which genes are being turned on or off).
    - This data allows scientists to map out not only which genes are expressed but also where they are expressed in the tissue, preserving the spatial structure of the tissue.
2. **Gene Expression**:
    - Gene expression refers to how much a specific gene is producing its product (RNA or protein). High expression means the gene is more active in that spot.
3. **Protein Expression**:
    - Proteins are often the functional products of genes. You will have data on protein levels for each of these spots, and your task is to predict these protein levels based on the gene expression in those same spots.
</aside>

- Problem is kind of single cell but actually in spots there might be multiple cells. I guess their combination is taken
- Gene and Protein data are sparse. Expression might be 0 which means it’s not expressed there.

We have the following data:

4 Different Sample:

- Spatial_CITEseq_Brain
- Spatial_CITEseq_Breast
- Spatial_CITEseq_Tonsil1
- Spatial_CITEseq_Tonsil2

Each has followings:

- filtered_feature_bc_matrix.h5
- spatial/aligned_fiducials.jpg
    
    ![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image.png)
    
- spatial/aligned_tissue_image.jpg
    
    ![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%201.png)
    
- spatial/barcode_fluorescence_intensity.csv
    
    ```
    barcode,in_tissue,DAPI_mean,DAPI_stdev,PCNA_mean,PCNA_stdev
    CGAGGATATTCAGAGC-1,0,NA,NA,NA,NA
    AGGATAGCGGACTATT-1,0,105.18811881188118,1.4468324657345573,113.02970297029702,2.3184812609374075
    ```
    
- spatial/cytassist_image.tiff
    
    ![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%202.png)
    
- spatial/detected_tissue_image.jpg
    
    ![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%203.png)
    
- spatial/scalefactors_json.json
    
    ```json
    {
      "regist_target_img_scalef": 0.21275087,
      "tissue_hires_scalef": 0.07091696,
      "tissue_lowres_scalef": 0.021275086,
      "fiducial_diameter_fullres": 323.039252660644,
      "spot_diameter_fullres": 184.593854628149
    }
    ```
    
- spatial/spatial_enrichment.csv
    
    ```
    Feature ID,Feature Name,Feature Type,I,P value,Adjusted p value,Feature Counts in Spots Under Tissue,Median Normalized Average Counts,Barcodes Detected per Feature,Feature Secondary Name
    BCL2,BCL2,Antibody Capture,0.9088464820450581,0.0,0.0,78707797,14636.515243609421,5756,BCL2
    PCNA,PCNA,Antibody Capture,0.8886609324609707,0.0,0.0,154194782,27263.14120741097,5756,PCNA
    ```
    
- spatial/tissue_hires_image.png
    
    ![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%204.png)
    
- spatial/tissue_lowres_image.png
    
    ![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%205.png)
    
- spatial/tissue_positions.csv
    
    ```
    barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
    CGAGGATATTCAGAGC-1,0,0,0,-4420,27760
    TCTGGTACTAATGCGG-1,0,0,2,-4414,27452
    ```
    
- spatial/tissue_positions_list.csv
    - Same with tissue_positions.csv

**How to find spot locations in the image?**

1. Resim boyutu
    1. Highres: (2000, 1744, 3)
    2. Lowres: (600, 523, 3)
    
    ```
    **Example Spatial Coordinates**
    [[23137. 10049.]
     [21933.  3952.]
     [ 6675. 10125.]]
    ```
    
    - Bu degerleri tissue_hires_scalef ile carparak asil pixel lokasyonlarini elde edebiliyoruz.
    
    ```
    **hires_scalefactor = 0.07091696
    lowres_scalefactor = 0.021275086
    
    hires_coords = spatial_coords * hires_scalefactor
    lowres_coords = spatial_coords * lowres_scalefactor**
    
    Spatial Coords:
     [[23137. 10049.]
     [21933.  3952.]
     [ 6675. 10125.]]
     
    High Resolution Image Coords:
     [[1640.80570352  712.64453104]
     [1555.42168368  280.26382592]
     [ 473.370708    718.03422   ]]
    ```
    

![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%206.png)

![image.png](Sp%20Trs%20Draft%20Notes%20120cb26dd6568085bcc9e2791d20c20e/image%207.png)
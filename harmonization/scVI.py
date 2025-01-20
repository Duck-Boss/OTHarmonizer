import scanpy as sc
import scarches as sca
import time

def reduce_batch_effect(
    adata,
    batch_index,
    annotation_index,
    layers_num=2,
    epoch_num=80,
    n_hidden=128,
    n_latent=10,
    n_layers=1,
    dropout_rate=0.1,
    dispersion='gene',
    gene_likelihood='zinb',
    latent_distribution='normal',
    **kwargs
):
    """
    Reduce batch effects using the SCVI model.

    Parameters:
        adata: AnnData object containing single-cell data.
        batch_index: Column name for batch information.
        layers_num: Number of layers in the SCVI model (default 2).
        epoch_num: Maximum number of training epochs (default 80).
        n_hidden: Number of neurons in the hidden layer (default 128).
        n_latent: Dimensionality of the latent space (default 10).
        n_layers: Number of layers in the encoder and decoder (default 1).
        dropout_rate: Dropout rate (default 0.1).
        dispersion: Dispersion model (default 'gene').
        gene_likelihood: Gene expression likelihood model (default 'zinb').
        latent_distribution: Type of latent distribution (default 'normal').
        **kwargs: Additional arguments passed to SCVI.
    """
    start_time = time.time()

    # Set up AnnData object
    sca.models.SCVI.setup_anndata(adata, batch_key=batch_index)

    # Initialize SCVI model
    vae = sca.models.SCVI(
        adata,
        n_hidden=n_hidden,
        n_latent=n_latent,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        dispersion=dispersion,
        gene_likelihood=gene_likelihood,
        latent_distribution=latent_distribution,
        **kwargs
    )

    # Train the model
    vae.train(max_epochs=epoch_num)

    # Calculate runtime
    end_time = time.time()
    print(f"Training completed. Time elapsed: {end_time - start_time:.2f} seconds")

    latent = sc.AnnData(vae.get_latent_representation())
    latent.obs[annotation_index] = adata.obs[annotation_index].tolist()
    latent.obs[batch_index] = adata.obs[batch_index].tolist()
    return latent




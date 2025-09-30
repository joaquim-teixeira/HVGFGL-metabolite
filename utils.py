
def sample_neg_mult_gamma_cont (n,p,size):
    p0 = 1-p.sum(axis=-1)
    #try:
    #    n_sample = torch.distributions.Gamma(n,1/(1/p0-1)).rsample([size])
    #except:
    n_sample = torch.distributions.Gamma(n,1/(1/(p0+ 1e-40)-1) + 1e-40).rsample([size])
    val = (n_sample/((1-p0+1e-40))).unsqueeze(-1)*p.unsqueeze(0)
    tst=torch.distributions.Normal(val, torch.sqrt(val+1e-40)).rsample()
    return(tst)

def find_neighbors(points, threshold):
    tree = KDTree(points)
    neighbors = tree.query_ball_tree(tree, threshold)
    return neighbors

def create_adjacency_matrix(points, neighbors):
    n = len(points)
    adjacency_matrix = np.zeros((n, n), dtype=int)

    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            adjacency_matrix[i][neighbor] = 1
            adjacency_matrix[neighbor][i] = 1
    return adjacency_matrix
def columnwise_corr(A, B, eps=1e-8):
    # A and B should be shape (n, p)
    A_mean = A.mean(dim=0)
    B_mean = B.mean(dim=0)

    A_centered = A - A_mean
    B_centered = B - B_mean

    numerator = (A_centered * B_centered).sum(dim=0)
    denom = (A_centered.square().sum(dim=0).sqrt() * B_centered.square().sum(dim=0).sqrt())

    corr = numerator / (denom + eps)  # eps for numerical stability
    return corr


def perlin_vein_pattern(size, scale=10.0, octaves=6, threshold=0.1):
    """
    Generate a vein-like pattern using Perlin noise and a threshold.

    Args:
        size (tuple): Height and width of the pattern.
        scale (float): Controls frequency of noise.
        octaves (int): Number of noise layers to sum.
        threshold (float): Narrow band around zero-crossing to define veins.

    Returns:
        np.array: Smoothed vein pattern.
    """
    H, W = size
    pattern = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            n = pnoise2(x / scale, y / scale, octaves=octaves)
            if abs(n) < threshold:  # narrow band around zero crossing
                pattern[y, x] = 1.0
    # Smooth the pattern to make veins more natural
    return gaussian_filter(pattern, sigma=1.2)


def sample_neg_mult_gamma_cont(n, p, size):
    """
    Approximate negative multinomial sampling using gamma and normal distributions.

    Args:
        n (Tensor): Gamma shape parameter.
        p (Tensor): Probabilities of categories.
        size (int): Number of samples.

    Returns:
        Tensor: Sampled counts (continuous approximation).
    """
    p0 = 1 - p.sum(axis=-1)
    n_sample = torch.distributions.Gamma(n, 1/(1/(p0 + 1e-40)-1) + 1e-40).rsample([size])
    val = (n_sample / (1 - p0 + 1e-40)).unsqueeze(-1) * p.unsqueeze(0)
    tst = torch.distributions.Normal(val, torch.sqrt(val + 1e-40)).rsample()
    return tst


def find_neighbors(points, threshold):
    """
    Use KDTree to find neighbors within a distance threshold.

    Args:
        points (ndarray): Coordinates of points.
        threshold (float): Distance threshold for neighbors.

    Returns:
        list: List of neighbor indices for each point.
    """
    tree = KDTree(points)
    neighbors = tree.query_ball_tree(tree, threshold)
    return neighbors


def create_adjacency_matrix(points, neighbors):
    """
    Convert neighbor list to adjacency matrix (symmetric).

    Args:
        points (ndarray): Points coordinates.
        neighbors (list): Neighbor indices from KDTree.

    Returns:
        ndarray: NxN adjacency matrix of 0/1 connections.
    """
    n = len(points)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            adjacency_matrix[i][neighbor] = 1
            adjacency_matrix[neighbor][i] = 1
    return adjacency_matrix


def multinomial_rvs(count, p):
    """
    Sample counts from a multinomial distribution.

    Args:
        count (ndarray): Total counts per sample.
        p (ndarray): Probability vectors per sample.

    Returns:
        ndarray: Multinomial samples per category.
    """
    out = np.zeros(p.shape, dtype=int)
    count = count.copy()
    ps = p.cumsum(axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

def sample_neg_mult_gamma_cont_cens(n, p, size, cens):
    """
    Sample from a negative multinomial-like distribution using a continuous gamma approximation,
    accounting for censoring.

    Args:
        n (Tensor): Shape parameter for the gamma distribution.
        p (Tensor): Probability vector for categories.
        size (int): Number of samples to generate.
        cens (Tensor): Censoring mask (1 if observed, 0 if censored).

    Returns:
        Tensor: Sampled values of shape [size, ...] matching input dimensions.
    """
    # Compute adjusted probability of zero-count based on censoring
    p0 = 1 - (p * cens).sum(axis=-1)

    # Sample latent counts from a gamma distribution approximating negative multinomial
    # The transformation 1/(1/(p0) - 1) is a parameterization trick to get desired mean
    n_sample = torch.distributions.Gamma(n, 1/(1/(p0 + 1e-40) - 1) + 1e-40).rsample([size])

    # Scale by censored probabilities and reshape to match dimensions
    val = (n_sample / (1 - p0 + 1e-40)).unsqueeze(-1) * (p * cens).unsqueeze(0)

    # Introduce Gaussian noise to approximate continuous counts
    tst = torch.distributions.Normal(val, torch.sqrt(val + 1e-40)).rsample()

    return tst


def elbo_lasso_cens(theta_q_loc, x, lam, adj, cens, lods, cr, samp_size, n_samp,
                    cens_exp, gamma_1_raw, gamma_2_raw, lam_1_raw, lam_2_raw,
                    dat_exp, a1, a2, cens_inf):
    """
    Computes the Evidence Lower Bound (ELBO) for a Hierarchical Variational Graph-Fused Lasso (HVGFL)
    model with censored negative multinomial data.

    Args:
        theta_q_loc (Tensor): Variational mean for theta logits.
        x (Tensor): Observed data.
        lam (float): Hyperparameter for gamma prior on lambda.
        adj (Tensor): Adjacency pairs for graph-fused lasso.
        cens (Tensor): Censoring mask.
        lods (Tensor): Limit of detection values for censored observations.
        cr (Tensor): Mask for completely observed data.
        samp_size (int): Number of negative multinomial samples per observation.
        n_samp (int): Number of variational samples.
        cens_exp (Tensor): Mask for censored entries.
        gamma_1_raw, gamma_2_raw (Tensor): Raw parameters for gamma variational distributions.
        lam_1_raw, lam_2_raw (Tensor): Raw parameters for gamma variational distributions of lambda.
        dat_exp (Tensor): Observed counts or exposures.
        a1, a2 (Tensor): Indices used to accumulate sums for precision computation.
        cens_inf (Tensor): Large constant to stabilize softmax for censored data.

    Returns:
        Tensor: Scalar ELBO value.
    """

    # Construct variational distributions for local and global scale parameters
    q_lam = torch.distributions.Gamma(torch.exp(lam_1_raw), torch.exp(lam_2_raw))
    lam_samp = q_lam.rsample([n_samp])

    q_gamma = torch.distributions.Gamma(torch.exp(gamma_1_raw), torch.exp(gamma_2_raw))
    q_gamma_samp = q_gamma.rsample([n_samp])

    # Determine shapes and prepare accumulators
    p1, E, q = q_gamma_samp.shape
    N = adj.max().item() + 1
    sums = torch.zeros(p1, N, q, dtype=torch.float32, device=device)

    # Compute sum of inverse gamma samples for precision of variational theta
    sums.scatter_add_(1, a1, 1 / q_gamma_samp)
    sums.scatter_add_(1, a2, 1 / q_gamma_samp)

    # Construct variational normal distribution for theta logits
    theta_q = torch.distributions.Normal(theta_q_loc, (1 / sums).sqrt())
    samps_q = theta_q.rsample([1]).squeeze(0)

    # Apply softmax to get probabilities
    theta = torch.softmax(samps_q, dim=-1)

    # Sample censored negative multinomial values for non-completely observed data
    wot = sample_neg_mult_gamma_cont_cens(dat_exp.sum(axis=-1)[:, ~cr],
                                          theta[:, ~cr, :],
                                          samp_size,
                                          cens_exp[:, ~cr, :])

    # Soft indicator for censoring likelihood contribution
    soft_indicator = torch.sigmoid(1e5 * (lods[~cr] - wot))
    vals = (((torch.sigmoid(1e5 * (soft_indicator.mean(axis=-1) - 1))) * 2.).mean(axis=0) + 1e-90).log()

    # Compute log-likelihood for observed (non-censored) data
    log_theta = samps_q - torch.logsumexp(samps_q - cens_inf, axis=-1).unsqueeze(-1)
    log_likelihood = (dat_exp[~cens_exp] * log_theta[~cens_exp]).sum() / n_samp
    log_likelihood = log_likelihood + vals.sum(axis=1).mean()

    # Graph-fused lasso prior: encourage neighboring differences to be small
    mu = (samps_q[:, adj[:, 0], :] - samps_q[:, adj[:, 1], :]).unsqueeze(0)
    laplace_prior = -(((mu.abs() * (1 / q_gamma_samp.unsqueeze(1))).sum([-1, -2])).mean())

    # Local gamma prior contribution
    local_prior = -(q_gamma_samp / lam_samp.unsqueeze(1)).sum([-1, -2]).mean()

    # Entropy terms for variational distributions
    ent_gamma = q_gamma.log_prob(q_gamma_samp).sum([-1, -2]).mean()
    ent_norm = theta_q.log_prob(samps_q).sum([-1, -2]).mean()

    # KL divergence between variational and prior for lambda
    lam_prior = torch.distributions.Gamma(1, lam)
    lam_kl = torch.distributions.kl_divergence(q_lam, lam_prior)

    return log_likelihood + laplace_prior + local_prior - ent_norm - ent_gamma - lam_kl.sum()

def elbo_lasso_cens_simple_MF(theta_q_loc, x, lam, adj, cens, lods, cr, samp_size, n_samp,
                              cens_exp, lam_1_raw, lam_2_raw, dat_exp, cens_inf, theta_q_sd):
    """
    Computes the Evidence Lower Bound (ELBO) for a Mean-Field Variational Inference
    version of the Graph-Fused Lasso model with censored negative multinomial data.

    Args:
        theta_q_loc (Tensor): Variational mean for theta logits.
        x (Tensor): Observed data (counts).
        lam (float or Tensor): Hyperparameter for the Gamma prior on lambda.
        adj (Tensor): Adjacency pairs for graph-fused lasso penalty.
        cens (Tensor): Mask indicating censored observations.
        lods (Tensor): Limit-of-detection values for censored entries.
        cr (Tensor): Mask for completely observed data points.
        samp_size (int): Number of negative multinomial samples for censored data.
        n_samp (int): Number of Monte Carlo samples from variational distributions.
        cens_exp (Tensor): Expanded censoring mask aligned with dat_exp.
        lam_1_raw (Tensor): Raw variational parameter for lambda shape (on log scale).
        lam_2_raw (Tensor): Raw variational parameter for lambda rate (on log scale).
        dat_exp (Tensor): Expanded observed data tensor (replicated for sampling).
        cens_inf (Tensor): Large constant for stabilizing softmax under censoring.
        theta_q_sd (Tensor): Log standard deviation parameter for variational Normal over logits.

    Returns:
        Tensor: Scalar ELBO estimate.
    """

    q_lam = torch.distributions.Gamma(torch.exp(lam_1_raw), torch.exp(lam_2_raw))
    lam_samp = q_lam.rsample([n_samp])

    theta_q = torch.distributions.Normal(theta_q_loc, torch.exp(theta_q_sd))
    samps_q = theta_q.rsample([n_samp])
    theta = torch.softmax(samps_q, dim=-1)

    wot = sample_neg_mult_gamma_cont_cens(dat_exp.sum(axis=-1)[:, ~cr],
                                          theta[:, ~cr, :],
                                          samp_size,
                                          cens_exp[:, ~cr, :])
    soft_indicator = torch.sigmoid(1e5 * (lods[~cr] - wot))
    vals = (((torch.sigmoid(1e5 * (soft_indicator.mean(axis=-1) - 1))) * 2.).mean(axis=0) + 1e-90).log()

    log_theta = samps_q - torch.logsumexp(samps_q - cens_inf, axis=-1).unsqueeze(-1)
    log_likelihood = (dat_exp[~cens_exp] * log_theta[~cens_exp]).sum() / n_samp
    log_likelihood = log_likelihood + vals.sum(axis=1).mean()

    mu = (samps_q[:, adj[:, 0], :] - samps_q[:, adj[:, 1], :]).unsqueeze(0)
    laplace_prior = -(((mu.abs() * (1 / lam_samp.unsqueeze(1))).sum([-1, -2])).mean())

    ent_norm = theta_q.log_prob(samps_q).sum([-1, -2]).mean()

    lam_prior = torch.distributions.Gamma(1, lam)
    lam_kl = torch.distributions.kl_divergence(q_lam, lam_prior)

    return log_likelihood + laplace_prior - ent_norm - lam_kl.sum()


def elbo_lasso_cens_MF(theta_q_loc, x, lam, adj, cens, lods, cr, samp_size, n_samp, cens_exp, gamma_1_raw, gamma_2_raw, lam_1_raw, lam_2_raw, dat_exp, cens_inf,theta_q_sd):

    q_lam = torch.distributions.Gamma(torch.exp(lam_1_raw), torch.exp(lam_2_raw))
    lam_samp = q_lam.rsample([n_samp])
    q_gamma = torch.distributions.Gamma(torch.exp(gamma_1_raw), torch.exp(gamma_2_raw))
    q_gamma_samp = q_gamma.rsample([n_samp])


    theta_q = torch.distributions.Normal(theta_q_loc, torch.exp(theta_q_sd))
    samps_q = theta_q.rsample([n_samp])
    theta = torch.softmax(samps_q,dim=-1)
    wot = sample_neg_mult_gamma_cont_cens(dat_exp.sum(axis=-1)[:,~cr], theta[:,~cr,:], samp_size, cens_exp[:,~cr,:])
    soft_indicator = torch.sigmoid(1e5 * (lods[~cr] - wot))
    vals = (((torch.sigmoid(1e5 * (soft_indicator.mean(axis=-1) - 1))) * 2.).mean(axis=0) + 1e-90).log()

    log_theta = samps_q - torch.logsumexp(samps_q-cens_inf, axis=-1).unsqueeze(-1)
    log_likelihood = (dat_exp[~cens_exp] * log_theta[~cens_exp]).sum() / n_samp


    log_likelihood = log_likelihood + vals.sum(axis=1).mean()

    mu = (samps_q[:, adj[:, 0], :] - samps_q[:, adj[:, 1], :]).unsqueeze(0)
    laplace_prior = -(((mu.abs()* (1 / q_gamma_samp.unsqueeze(1))).sum([-1, -2])).mean())
    local_prior = -(q_gamma_samp / lam_samp.unsqueeze(1)).sum([-1, -2]).mean()

    ent_gamma = q_gamma.log_prob(q_gamma_samp).sum([-1, -2]).mean()
    ent_norm = theta_q.log_prob(samps_q).sum([-1, -2]).mean()

    lam_prior = torch.distributions.Gamma(1, lam)
    lam_kl = torch.distributions.kl_divergence(q_lam, lam_prior)
    return log_likelihood + laplace_prior + local_prior - ent_norm - ent_gamma - lam_kl.sum()

clear; clc; close all;
rng(0);


%% Problem setup
m = 50;
n = 30;
A = randn(m,n);
b = rand(m,1) + 0.5;


%% Build weighted graph Laplacian (connected graph)
W = zeros(n);

for i = 1:n-1
    W(i,i+1) = 0.5 + rand();  
    W(i+1,i) = W(i,i+1);
end

extra_edges = round(n); 
for k = 1:extra_edges
    i = randi(n); j = randi(n);
    if i ~= j
        W(i,j) = W(i,j) + 0.2*rand();
        W(j,i) = W(i,j);
    end
end
W(1:n+1:end) = 0;          
d = sum(W,2);
L = diag(d) - W;
L = (L + L')/2;            

fprintf('=== Laplacian Properties ===\n');
fprintf('Symmetry ||L-L''||_F = %.3e\n', norm(L-L','fro'));
e = eig(full(L));
fprintf('Min eigenvalue  = %.3e (should be ~0)\n', min(e));
fprintf('Rank(L)         = %d (should be n-1=%d for connected graph)\n', rank(full(L)), n-1);
fprintf('L*1             = %.3e (should be ~0)\n\n', norm(L*ones(n,1)));


%% CVX solve LR-NNLS
cvx_begin quiet
    variables x(n) y(m)
    dual variables lambda mu
    minimize( 0.5*sum_square(y) + 0.5*quad_form(x, L) )
    subject to
        lambda : y == A*x - b;
        mu     : x >= 0;
cvx_end

fprintf('=== CVX Solution ===\n');
fprintf('Status: %s\n', cvx_status);
fprintf('Optimal value: %.6e\n\n', cvx_optval);


%% Verify KKT conditions
fprintf('=== KKT Conditions Verification ===\n');

% Stationarity: Lx* + A^T*lambda* - mu* = 0
kkt_stationarity = norm(L*x + A'*lambda - mu);
fprintf('Stationarity ||Lx* + A^T lambda* - mu*||  = %.3e\n', kkt_stationarity);

% Complementary slackness: mu^T * x = 0
comp_slack = abs(mu' * x);
fprintf('Complementarity |mu*^T x*|                = %.3e\n', comp_slack);

% Primal feasibility
fprintf('Nonnegativity min(x*)                     = %.3e\n', min(x));
fprintf('Constraint ||Ax* - b - y*||               = %.3e\n\n', norm(A*x - b - y));


%% Compute the pseudoinverse reconstruction
Ldag = pinv(L);
v = mu - A'*lambda;    
z = Ldag * v;              


%% Method 1: Determine c from components where x*_i = 0 (Active constraints)
fprintf('=== Method 1: Determine c from Zero Components ===\n');

% Find indices where x*_i = 0 (equivalently, where mu*_i > 0)
tol_zero = 1e-6;
zero_indices = find(mu > tol_zero);  % Active constraints: x*_i = 0

fprintf('Components where x*_i = 0: %d out of %d\n', length(zero_indices), n);

if ~isempty(zero_indices)
    % For indices where x*_i = 0, we have z_i + c = 0, so c = -z_i
    c_from_zeros = -z(zero_indices);
    c_consistency = std(c_from_zeros);
    c_method1 = mean(c_from_zeros);
    
    fprintf('Consistency check: std(c from zeros) = %.3e\n', c_consistency);
    fprintf('Determined c = %.6e\n', c_method1);
    
    % Additional check: verify all zero components give similar c
    if c_consistency < 1e-5
        fprintf('All zero components yield consistent c\n');
    else
        fprintf('Warning: inconsistency detected (std = %.3e)\n', c_consistency);
    end
else
    % No components at zero (i.e., mu* = 0), can choose c = 0
    c_method1 = 0;
    fprintf('No zero components (mu* = 0), setting c = 0\n');
end

% Reconstruct x using Method 1
x_method1 = z + c_method1 * ones(n,1);
error_m1 = norm(x - x_method1);
rel_error_m1 = error_m1 / norm(x);

fprintf('Reconstruction error ||x* - (z + c*1)||   = %.3e\n', error_m1);
fprintf('Relative error                            = %.3e\n', rel_error_m1);
fprintf('min(x_reconstructed)                      = %.3e\n\n', min(x_method1));


%% Method 2: Least squares fit for c (for comparison); 
%Find c to minimize ||x - z - c*1||^2; 
%Solution: c = mean(x - z)

fprintf('=== Method 2: Least Squares Fit (for comparison) ===\n');

c_method2 = mean(x - z);
fprintf('c (from least squares) = %.6e\n', c_method2);

x_method2 = z + c_method2 * ones(n,1);
error_m2 = norm(x - x_method2);
rel_error_m2 = error_m2 / norm(x);

fprintf('Reconstruction error    = %.3e\n', error_m2);
fprintf('Relative error          = %.3e\n', rel_error_m2);

% Compare the two methods
fprintf('\nComparison: |c_method1 - c_method2| = %.3e\n', abs(c_method1 - c_method2));
if abs(c_method1 - c_method2) < 1e-6
    fprintf('Both methods yield the same c\n\n');
else
    fprintf('Methods give different c values\n\n');
end


%% Method 3: Without c (baseline)
fprintf('=== Baseline: Without c (to show necessity) ===\n');

x_baseline = z;  % No c*1 term
error_baseline = norm(x - x_baseline);
rel_error_baseline = error_baseline / norm(x);

fprintf('Reconstruction error ||x* - z||  = %.3e\n', error_baseline);
fprintf('Relative error                   = %.3e\n', rel_error_baseline);
fprintf('Error amplification factor       = %.1f\n\n', error_baseline/error_m1);


%% Summary table
fprintf('================================================================================\n');
fprintf('                         RECONSTRUCTION SUMMARY                                 \n');
fprintf('================================================================================\n');
fprintf(' Method                     | Abs Error    | Rel Error    | Min Value         \n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf(' Zero Components (Method 1) | %.4e | %.4e | %+.4e      \n', error_m1, rel_error_m1, min(x_method1));
fprintf(' Least Squares (Method 2)   | %.4e | %.4e | %+.4e      \n', error_m2, rel_error_m2, min(x_method2));
fprintf(' Without c (Baseline)       | %.4e | %.4e | %+.4e      \n', error_baseline, rel_error_baseline, min(x_baseline));
fprintf('================================================================================\n\n');


%% Final verification checklist
fprintf('================================================================================\n');
fprintf('                     THEORETICAL VALIDATION CHECKLIST                           \n');
fprintf('================================================================================\n');
fprintf(' 1. KKT stationarity satisfied?       %s (residual = %.3e)            \n', ...
    iff(kkt_stationarity < 1e-5, '[YES]', '[NO] '), kkt_stationarity);
fprintf(' 2. Complementary slackness holds?    %s (|mu^T x| = %.3e)            \n', ...
    iff(comp_slack < 1e-5, '[YES]', '[NO] '), comp_slack);
if ~isempty(zero_indices)
    fprintf(' 3. Zero components consistent?       %s (std(c) = %.3e)             \n', ...
        iff(std(c_from_zeros)<1e-5, '[YES]', '[NO] '), std(c_from_zeros));
else
    fprintf(' 3. Zero components consistent?       [N/A] (no zero components)              \n');
end
fprintf(' 4. Method 1 reconstruction accurate? %s (rel_err = %.3e)            \n', ...
    iff(rel_error_m1 < 1e-5, '[YES]', '[NO] '), rel_error_m1);
fprintf(' 5. Method 1 = Method 2?              %s (diff = %.3e)               \n', ...
    iff(abs(c_method1-c_method2) < 1e-6, '[YES]', '[NO] '), abs(c_method1-c_method2));
fprintf(' 6. c is necessary?                   %s (baseline %.0fx worse)         \n', ...
    iff(error_baseline > 10*error_m1, '[YES]', '[NO] '), error_baseline/error_m1);
fprintf('================================================================================\n');

function s = iff(cond, true_str, false_str)
    if cond
        s = true_str;
    else
        s = false_str;
    end
end
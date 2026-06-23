function [ vertsTransformed, X ] = nricp_landmarks( Source, Target, Options, ls_source, ls_target)
% nricp_landmarks (modified) - Non-rigid ICP with support for point clouds.
% This modified version works also if Source and/or Target have no 'faces'
% field (i.e., are point clouds). It does not compute faces in output.
%
% Inputs/Outputs: same as original. See your original header for details.
%
% Main changes:
% - if faces absent, build adjacency from k-NN graph (for stiffness term)
% - estimate normals if missing (local PCA)
% - projectNormals supports both mesh-target (intersection) and
%   pointcloud-target (nearest along normal)
% - find_bound disabled if no faces present

% --------- Parameters for fallbacks (tunable) ----------
knn_stiffness = 6;   % edges per vertex for adjacency when faces missing
knn_normal = 16;     % neighbors to estimate normals
knn_project = 20;    % neighbors to consider when projecting along normal
% -------------------------------------------------------

% Set default parameters (same as original)
if ~isfield(Options, 'gamm'); Options.gamm = 1; end
if ~isfield(Options, 'epsilon'); Options.epsilon = 1e-4; end
if ~isfield(Options, 'lambda'); Options.lambda = 1; end
if ~isfield(Options, 'alphaSet'); Options.alphaSet = linspace(100, 10, 20); end
if ~isfield(Options, 'biDirectional'); Options.biDirectional = 0; end
if ~isfield(Options, 'useNormals'); Options.useNormals = 0; end
if ~isfield(Options, 'plot'); Options.plot = 0; end
if ~isfield(Options, 'rigidInit'); Options.rigidInit = 1; end
if ~isfield(Options, 'ignoreBoundary'); Options.ignoreBoundary = 0; end
if ~isfield(Options, 'normalWeighting'); Options.normalWeighting = 0; end
if ~isfield(Options, 'landmarks'); Options.landmarks = 0; end
if ~isfield(Options, 'betaSet'); Options.betaSet = linspace(3, 0, 20); end

% Optionally plot source and target surfaces/pointclouds
if Options.plot == 1
    clf;
    % plot Target: if faces exist plot patch, otherwise scatter3
    if isfield(Target,'faces') && ~isempty(Target.faces)
        PlotTarget = rmfield_keep(Target,'normals');
        p = patch(PlotTarget, 'facecolor', 'b', 'EdgeColor',  'none', ...
                  'FaceAlpha', 0.5);
    else
        scatter3(Target.vertices(:,1), Target.vertices(:,2), Target.vertices(:,3), 6, [0 0 1]);
        hold on;
    end

    % plot Source
    if isfield(Source,'faces') && ~isempty(Source.faces)
        PlotSource = rmfield_keep(Source,'normals');
        h = patch(PlotSource, 'facecolor', 'r', 'EdgeColor',  'none', ...
            'FaceAlpha', 0.5);
    else
        h = scatter3(Source.vertices(:,1), Source.vertices(:,2), Source.vertices(:,3), 6, [1 0 0]);
    end

    material dull; light; grid on; xlabel('x'); ylabel('y'); zlabel('z');
    view([60,30]); axis equal; axis manual;
    legend('Target', 'Source', 'Location', 'best')
    drawnow;
end

% Get source vertices 
vertsSource = Source.vertices;
nVertsSource = size(vertsSource, 1);

% Get target vertices
vertsTarget = Target.vertices;

% Estimate normals if missing
if isfield(Source,'normals') && ~isempty(Source.normals)
    normalsSource = Source.normals;
else
    normalsSource = estimate_normals(vertsSource, knn_normal);
end

if isfield(Target,'normals') && ~isempty(Target.normals)
    normalsTarget = Target.normals;
else
    normalsTarget = estimate_normals(vertsTarget, knn_normal);
end

% If Options.normalWeighting == 0 later code will not use normals

% If biDirectional sample points from Target if needed
if Options.biDirectional == 1
    samplesTarget = sampleVerts_struct(Target, 15);
    nSamplesTarget = size(samplesTarget, 1);
end

% Set matrix G (equation (3) in Amberg et al.) 
G = diag([1 1 1 Options.gamm]);

% Build incidence matrix M:
if isfield(Source,'faces') && ~isempty(Source.faces)
    A = triangulation2adjacency(Source.faces, Source.vertices);
else
    % Build adjacency using k-NN graph on vertices (symmetric)
    idx = knnsearch(vertsSource, vertsSource, 'K', knn_stiffness+1); % includes self
    A = sparse(nVertsSource, nVertsSource);
    for i=1:nVertsSource
        neigh = idx(i,2:end); % skip self
        A(i, neigh) = 1;
        A(neigh, i) = 1;
    end
    % ensure no self loops
    A = A - spdiags(diag(A),0,size(A,1),size(A,2));
end

M = adjacency2incidence(A)' ;

% Precompute kronecker product of M and G
kron_M_G = kron(M, G);

% Set matrix D (equation (8) in Amberg et al.)
I = (1:nVertsSource)';
J = 4*I;
D = sparse([I;I;I;I],[J-3;J-2;J-1;J],[vertsSource(:);ones(nVertsSource,1)],nVertsSource, 4*nVertsSource);

% Set Dl and Ul if landmarks present
if Options.landmarks == 1
    nVertsLs = length(ls_source);
    Dl = sparse(nVertsLs, 4*nVertsSource);
    for j = 1:nVertsLs
        cor = ls_source(j);
        Dl(j,(cor*4-3):(cor*4)) = [vertsSource(cor,:) 1];
    end
    Ul = vertsTarget(ls_target, :);
end

% Set weights vector
wVec = ones(nVertsSource,1);

% Get boundary vertex indices on target surface if required.
if Options.ignoreBoundary == 1
    if isfield(Target,'faces') && ~isempty(Target.faces)
        bdr = find_bound(vertsTarget, Target.faces);
    else
        warning('ignoreBoundary option requested but Target has no faces. Ignoring boundary option.');
        bdr = [];
    end
end

% Set target points matrix tarU and target weights matrix tarU if bidirectional
if Options.biDirectional == 1
    tarU = samplesTarget;
    tarW = eye(nSamplesTarget);
end

% Do rigid ICP if requested
if Options.rigidInit == 1
    disp('* Performing rigid ICP...');
    if Options.ignoreBoundary == 0
        bdr = 0;
    end
    [R, t] = icp(vertsTarget', vertsSource', 50, 'Verbose', true, ...
                 'EdgeRejection', logical(Options.ignoreBoundary), ...
                 'Boundary', bdr', 'Matching', 'kDtree');
    X = repmat([R'; t'], nVertsSource, 1);
    vertsTransformed = D*X;
    if Options.plot == 1
        if isfield(h,'Vertices')   % patch
            set(h, 'Vertices', vertsTransformed);
        else
            % scatter handle: replot
            delete(h);
            h = scatter3(vertsTransformed(:,1), vertsTransformed(:,2), vertsTransformed(:,3), 6, [1 0 0]);
        end
        drawnow;
    end
else
    X = repmat([eye(3); [0 0 0]], nVertsSource, 1);
end

% Outer loop over stiffness parameters
nAlpha = numel(Options.alphaSet);
disp('* Performing non-rigid ICP...');
for i = 1:nAlpha
    alpha = Options.alphaSet(i);
    beta = Options.betaSet(i);
    oldX = 10*X;
    while norm(X - oldX) >= Options.epsilon
        vertsTransformed = D*X;
        if Options.plot == 1
            if isfield(h,'Vertices')
                set(h, 'Vertices', full(vertsTransformed));
            else
                delete(h);
                h = scatter3(vertsTransformed(:,1), vertsTransformed(:,2), vertsTransformed(:,3), 6, [1 0 0]);
            end
            drawnow;
        end
        
        % Update correspondences: closest points on target
        targetId = knnsearch(vertsTarget, vertsTransformed);
        U = vertsTarget(targetId,:);
        
        % ignoreBoundary handling (works only if bdr computed)
        if Options.ignoreBoundary == 1 && ~isempty(bdr)
            tarBoundary = ismember(targetId, bdr);
            wVec = ~tarBoundary;
        end
        
        % Normal weighting: compute transformed normals and compare with target normals
        if Options.normalWeighting == 1
            I = (1:nVertsSource)';
            J = 4*I;
            N = sparse([I;I;I;I],[J-3;J-2;J-1;J],[normalsSource(:);ones(nVertsSource,1)],nVertsSource, 4*nVertsSource);
            normalsTransformed = N*X;
            corNormalsTarget = normalsTarget(targetId,:);
            crossNormals = cross(corNormalsTarget, normalsTransformed);
            crossNormalsNorm = sqrt(sum(crossNormals.^2,2));
            dotNormals = dot(corNormalsTarget, normalsTransformed, 2);
            angle = atan2(crossNormalsNorm, dotNormals);
            wVec = wVec .* (angle<pi/4);
        end
        
        W = spdiags(wVec, 0, nVertsSource, nVertsSource);
        
        % Bidirectional source->target correspondences (if requested)
        if Options.biDirectional == 1
            transformedId = knnsearch(vertsTransformed, samplesTarget);
            tarD = sparse(nSamplesTarget, 4 * nVertsSource);
            for j = 1:nSamplesTarget
                cor = transformedId(j);
                tarD(j,(4 * cor-3):(4 * cor)) = [vertsSource(cor,:) 1];
            end
        end
        
        % Build system (A,B) as in original paper/code
        A = [ alpha .* kron_M_G;
              W * D;
            ];
        B = [ zeros(size(M,1)*size(G,1), 3);
              W * U;
            ];
        if Options.biDirectional == 1
            A = [A; Options.lambda .* tarW * tarD];
            B = [B; Options.lambda .* tarW * tarU];
        end
        if Options.landmarks == 1
            A = [A; beta .* Dl];
            B = [B; beta .* Ul];
        end
        
        % Solve for X
        oldX = X;
        X = (A' * A) \ (A' * B);
    end
end

% Final transformed points
vertsTransformed = D*X;

% If useNormals==1, project along normals; else snap to nearest points
if Options.useNormals == 1
    disp('* Projecting transformed points onto target along surface normals...');
    normalsTemplate = normalsSource;
    I = (1:nVertsSource)';
    J = 4*I;
    N = sparse([I;I;I;I],[J-3;J-2;J-1;J],[normalsTemplate(:);ones(nVertsSource,1)],nVertsSource, 4*nVertsSource);
    normalsTransformed = N*X;
    
    vertsTransformed = projectNormals_modified(vertsTransformed, Target, normalsTransformed, knn_project);
else
    targetId = knnsearch(vertsTarget, vertsTransformed);
    corTargets = vertsTarget(targetId,:);
    if Options.ignoreBoundary == 1 && ~isempty(bdr)
        tarBoundary = ismember(targetId, bdr);
        wVec = ~tarBoundary;
    end
    vertsTransformed(wVec,:) = corTargets(wVec,:);
end

% Update plot
if Options.plot == 1
    if isfield(h,'Vertices')
        set(h, 'Vertices', vertsTransformed);
    else
        delete(h);
        scatter3(vertsTransformed(:,1), vertsTransformed(:,2), vertsTransformed(:,3), 6, [0 1 0]);
    end
    drawnow;
    pause(2);
    if exist('p','var'); delete(p); end
end

end

% ----------------------- Helper functions -----------------------

function normals = estimate_normals(pts, k)
% Estimate normals by local PCA (smallest eigenvector)
n = size(pts,1);
normals = zeros(n,3);
idx = knnsearch(pts, pts, 'K', k+1); % include self
for i=1:n
    neigh = idx(i,2:end);
    P = pts(neigh,:) - mean(pts(neigh,:),1);
    C = (P'*P) / size(P,1);
    [V,~] = eig(C);
    nvec = V(:,1); % eigenvector with smallest eigenvalue
    normals(i,:) = nvec';
end
% orient normals consistently (simple propagation)
% pick a seed normal and orient neighbors to have positive dot
visited = false(n,1);
stack = 1;
visited(1)=true;
while ~isempty(stack)
    v = stack(end); stack(end)=[]; 
    neigh = knnsearch(pts, pts(v,:), 'K', 6);
    neigh = neigh(2:end);
    for u=neigh
        if ~visited(u)
            if dot(normals(v,:), normals(u,:)) < 0
                normals(u,:) = -normals(u,:);
            end
            visited(u)=true;
            stack(end+1)=u;
        end
    end
end
end

function out = sampleVerts_struct(Mesh, radius)
% same behaviour as original sampleVerts but accepts structs with/without faces
verts = Mesh.vertices;
out = sampleVerts(verts, radius);
end

function samples = sampleVerts(verts, radius)
samples = [];
vertsLeft = verts;
itt = 1;
while size(vertsLeft, 1) > 0
    nVertsLeft = size(vertsLeft, 1);
    vertN = randsample(nVertsLeft, 1);
    vert = vertsLeft(vertN, :);
    samples(itt,:) = vert;
    idx = rangesearch(vertsLeft, vert, radius);
    idRemove = idx{1};
    vertsLeft(idRemove, :) = [];
    itt = itt + 1;
end
end

function bound = find_bound(pts, poly)
% If faces/triangulation available, compute boundary; otherwise empty
if isempty(poly)
    bound = [];
    return;
end
poly = double(poly);
pts = double(pts);
TR = triangulation(poly, pts);
FF = freeBoundary(TR);
bound = FF(:,1);
end

function projections = projectNormals_modified(sourceVertices, Target, normals, knn_project)
% If Target has faces -> use intersectLineMesh3d as original.
% If not, approximate projection by searching along the normal among nearby points.
nVerticesSource = size(sourceVertices, 1);
projections = zeros(nVerticesSource, 3);
vertsT = Target.vertices;

useMesh = isfield(Target,'faces') && ~isempty(Target.faces);

if useMesh
    for i=1:nVerticesSource
        vertex = sourceVertices(i,:);
        normal = normals(i,:);
        line = createLine3d(vertex, normal(1), normal(2), normal(3));
        intersection = intersectLineMesh3d(line, Target.vertices, Target.faces);
        if ~isempty(intersection)
            [~,I] = min(sqrt(sum((intersection - repmat(vertex,size(intersection,1),1)).^2, 2)));
            projections(i,:) = intersection(I,:);
        else
            projections(i,:) = vertex;
        end
    end
else
    % target is point cloud: for each source vertex, examine k nearest target points
    idx = knnsearch(vertsT, sourceVertices, 'K', knn_project);
    for i=1:nVerticesSource
        v = sourceVertices(i,:);
        nrm = normals(i,:);
        neighIdx = idx(i,:);
        neighPts = vertsT(neighIdx,:);
        vecs = neighPts - repmat(v, size(neighPts,1), 1);
        t = vecs * nrm';   % projection scalar along normal (can be negative)
        % consider only points in direction of the normal (t>0)
        posMask = t > 1e-6;
        if any(posMask)
            cand = neighPts(posMask,:);
            vecsCand = vecs(posMask,:);
            % perpendicular distances:
            tn = t(posMask) .* repmat(nrm, sum(posMask), 1); 
            perpDist = sqrt(sum((vecsCand - tn).^2, 2));
            [~,mi] = min(perpDist);
            projections(i,:) = cand(mi,:);
        else
            % fallback to nearest neighbour
            projections(i,:) = neighPts(1,:);
        end
    end
end
end

function S = rmfield_keep(S, fname)
% helper: remove field if exists else return S
if isfield(S, fname)
    S = rmfield(S, fname);
end
end
using Flux, Statistics, LinearAlgebra, Plots, MAT, DelimitedFiles, Random, Distributed, SharedArrays, DistributedArrays, LaTeXStrings
using Flux: reset!
include("./../../DataUtilities.jl")
include("./../../Models.jl")
include("./../../TrainingUtilities.jl")

dt = 0.01

data = readdlm("./examples/Filter banks/HH_io.txt")
v = data[:,2]

Ti = 50
Tf = 500

u = v[round(Int,Ti/dt):round(Int,Tf/dt)]
E = sum(u)
t = dt*(0:length(u)-1)
plot(u)
##
δ = [1;zeros(length(u)-1)]
u = u
τ = [0.5*1.26^i for i=0:13] #[] [0.5;5;50;500]
# τ = [[0.5,0.75], [5, 7.5], [50,75], [500,750]]

# Create filters
FB_gobf = OrthogonalFilterCell(τ,dt)
FB_diag = DiagonalFilterCell(τ,dt)

# Impulse response
g_gobf = FB_gobf(δ,DC=false)
g_diag = FB_diag(δ,DC=false)

# Test the orthogonalizing state-space transformation
T = OrthogonalTransformation(FB_diag)
plot(t,(T*g_diag)',title="diag",linestyle=:dash,xlims=(0,10),legend=false)
plot!(t,g_gobf',title="gobf",legend=false)
norm(T*g_diag-g_gobf)

## Normalized impulse response
# normLayer = MinMaxNorm(g_gobf,-1,1)
normLayer = CenterNorm(g_gobf,-1,1)
g_gobf = normLayer(g_gobf)
display(sum(g_gobf[1,:].*g_gobf[2,:]))

# Spike response
y_gobf = FB_gobf(u,DC=true)
y_diag = FB_diag(u)

# normLayer = MinMaxNorm(y_gobf,-1,1)
normLayer = CenterNorm(y_gobf,-1,1)
y_gobf = normLayer(y_gobf)
# display(sum(y_gobf[2,:].*y_gobf[3,:]))

plot(plot(t,g_diag',title="diag"),plot(t,g_gobf',title="gobf"),layout=(2,1),xlims=(0,400))
# plot(plot(t,y_diag',title="diag"),plot(t,y_gobf',title="gobf"),layout=(2,1),xlims=(0,100))


#################
## Something else
#################
p_y_gobf_1 = plot(t,y_gobf[1,:],ylabel=L"x_1",yticks=[-1,-0.5,0,0.5,1],color=:blue)
p_y_gobf_2 = plot(t,y_gobf[2,:],ylabel=L"x_2",yticks=[-1,-0.5,0,0.5,1],color=:red)
p_y_gobf_3 = plot(t,y_gobf[3,:],ylabel=L"x_3",yticks=[-1,0,1],color=:green)
p_y_gobf_4 = plot(t,y_gobf[4,:],ylabel=L"x_4",yticks=[-1,0,1],color=:purple,xlabel="t [ms]")

p_y_diag_1 = plot(t,y_diag[1,:],ylabel=L"x_1",yticks=[-1,-0.5,0,0.5,1],color=:blue)
p_y_diag_2 = plot(t,y_diag[2,:],ylabel=L"x_2",yticks=[-1,-0.5,0,0.5,1],color=:red)
p_y_diag_3 = plot(t,y_diag[3,:],ylabel=L"x_3",yticks=[-1,-0.5,0,0.5,1],color=:green)
p_y_diag_4 = plot(t,y_diag[4,:],ylabel=L"x_3",yticks=[-1,-0.5,0,0.5,1],color=:purple,xlabel="t [ms]")

p_v = plot(t,u*inv.(maximum(u)),ylabel=L"v",yticks=[-1,-0.5,0,0.5,1])

plot(p_v,p_v,p_y_diag_2,p_y_gobf_2,p_y_diag_3,p_y_gobf_3,p_y_diag_4,p_y_gobf_4,layout=(4,2),
    legend=false,
    # xticks=[0,25,50],
    linewidth=5,
    xlabelfontsize=8,
    xlims=(0,35),
    grid=false,
    ticks=false,
    xlabel="",
    ylabel="",
    axis=false,
    # bottommargins=[-2Plots.mm -2Plots.mm -2Plots.mm -2Plots.mm -2Plots.mm],
    )
# savefig("./examples/Filter banks/responses.png")
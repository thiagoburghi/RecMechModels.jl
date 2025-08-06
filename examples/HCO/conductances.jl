include("./../../RecMechModels.jl")
datapath = string("examples/HCO/data/")
modelpath = string("examples/HCO/results/models/")
figurepath = string("examples/HCO/results/figures/")
@load string(modelpath,"best_models.bson") modelmech modelms modellump

using PlotlyJS
plotlyjs()

function spherical_eye(azimuth_deg, elevation_deg, r=2.0)
    ϕ = deg2rad(azimuth_deg)
    θ = deg2rad(elevation_deg)
    x = r * cos(θ) * sin(ϕ)
    y = r * cos(θ) * cos(ϕ)
    z = r * sin(θ)
    return (x=x, y=y, z=z)
end

###################################
## Frequency-dependent conductances
###################################
n,m = 2,2
V̄ = collect(-60:0.1:-35)
Ω = [5e-3:1e-3:1e-2; 1e-2:0.01:0.1; 0.1:0.1:1; 1:20] #; 10*pi

model_training_type = :mech

@eval model = $(Symbol("model", model_training_type))
# IVion,IVleak=IV(net.cell,V̄,V̄)
Y_v_ω=localAdmittances(model.cell,V̄,Ω)

# Find bifurcations
# res = findBifurcation(model,(n,m),Ω₀=[0.0,0.1,0.01,0.001],V₀=[-70,-65.0,-60.0,-55.0,-50,-45.,-40.0,-35.0,-30.0])

##
data_min = minimum(real.(Y_v_ω[n,m]))
data_max = maximum(real.(Y_v_ω[n,m]))
custom_gradient = cgrad([:red, :white, :blue],[0.0,(0.0 - data_min) / (data_max - data_min),1.0])
l = min(abs(data_min), abs(data_max))
cmin = -4
cmax = 1
colorscale = [
    (0.0, "rgb(100,0,0)"),   # Red
    (0.6, "rgb(255,0,0)"),   # Red
    (0.8, "rgb(200,200,200)"), # White
    (1.0, "rgb(0,0,255)")    # Blue
]

abs(data_min) > abs(data_max) ? l = abs(data_min) : l = abs(data_max)

colorbar = attr(
    len = 0.6,          # 40% of plot height
    thickness = 12,     # thin bar
    x = 1.02,           # slightly outside plot
    y = 0.5,            # centered vertically
    yanchor = "middle"
)

# Plot the frequency dependent conductance surface
pltf=PlotlyJS.surface(x=V̄,y=log10.(Ω),z=real.(Y_v_ω[n,m])',colorbar=colorbar,
            colorscale = colorscale,
            cmin = cmin,
            cmax = cmax,
            # color = custom_gradient,
            # xticks=[-50,-40,-30],
            # camera=(azimuth, elevation),
            # size=(800, 600),
            # yticks=(0.0:-1.0:-3.0, [round(2*pi ./ 10 .^ y,digits=1) for y in 0.0:-1.0:-3.0]),
            )

# Plot the speficied curves separately and on the surfaces
curves = []
freqplots = []
for freq_ind = [1,28]
    Ω[freq_ind]
    curve_trace = PlotlyJS.scatter3d(
        x = V̄,
        y = log10(Ω[freq_ind])*ones(length(V̄)),
        z = real.(Y_v_ω[n,m])[freq_ind,:],  # lifted slightly above surface
        mode = "lines",
        line = attr(color = "black", width = 8),
        showlegend = false
    )
    push!(curves,curve_trace)
    push!(freqplots,Plots.plot(V̄,real.(Y_v_ω[n,m])[freq_ind,:],
                            label=string("ω=",round(Int,Ω[freq_ind]),"rad/ms"),
                            linewidth=3,
                            color=:black,
                            legend=:bottomleft,
                            ylabel="G(v,ω) [nS]",
                            xlabel="v [mV]",
                            margins=5Plots.mm,
                            size=(400,200)))
    Plots.savefig(string(figurepath,"conductance/freq=",Ω[freq_ind],"_neuron_",n,".svg"))
end

# Plot the bifurcations
bifs = []
for i=1:length(res)
    closest_v, idx_v = findmin(abs.(V̄ .- res[i][:v]))
    closest_ω, idx_ω = findmin(abs.(Ω .- res[i][:ω]))

    star_trace = PlotlyJS.scatter3d(
        x = [res[i][:v]],
        y = [log10(res[i][:ω])],
        z = [real.(Y_v_ω[n,m])[idx_ω,idx_v]+0.05],
        mode = "markers",
        marker = attr(
            size = 1,                # make it big
            color = "black",           # or any color
            symbol = "star"           # star shape
        ),
        showlegend = false
    )
    push!(bifs,star_trace)
end

# Configure and create the plot
azimuth = 140  # Azimuthal angle
elevation = 50  # Elevation angle
layout = Layout(
    autosize=false,
    title=string("Frequency-dependent conductance of HCO neuron ",n),
    scene = attr(
        xaxis = attr(title = "v [mV]"),
        yaxis = attr(title = "log₁₀(ω) [rad/ms]"),
        zaxis = attr(title = "G(v,ω) [nS]",
        # range = [-4, 1],
        ),
        camera = attr(eye = spherical_eye(azimuth, elevation, 2.1)),
        scene_camera_eye=attr(x=5, y=0.88, z=-0.64),
        aspectratio = attr(x = 1, y = 1, z = 0.5),
    ),
    #width=500, 
    # height=500,
    margin=attr(l=0, r=0, b=20, t=0)  # Increased bottom margin to prevent cutting
)
plt = PlotlyJS.plot([pltf,curves...,bifs[1:3]...], layout)#
PlotlyJS.savefig(plt,string(figurepath,"conductance/n=",n,"m=",m,"bifs.svg"))
plt
##

##
freqplots = []


PlotlyJS.plot(pltf)
pltf
include("./../../RecMechModels.jl")
datapath = string("examples/HCO/data/")
modelpath = string("examples/HCO/results/models/")
figurepath = string("examples/HCO/results/figures/")
@load string(modelpath,"best_models.bson") modelmech modelms

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
V̄ = collect(-60:0.1:-35)
Ω = [1e-2:0.01:0.1; 0.1:0.1:1; 1:30; 10*pi]

model_training_type = :mech

@eval model = $(Symbol("model", model_training_type))
# IVion,IVleak=IV(net.cell,V̄,V̄)
Y_v_ω=localAdmittances(model.cell,V̄,Ω)


##
n,m = 1,1
data_min = minimum(real.(Y_v_ω[n,m]))
data_max = maximum(real.(Y_v_ω[n,m]))
custom_gradient = cgrad([:red, :white, :blue],[0.0,(0.0 - data_min) / (data_max - data_min),1.0])
l = min(abs(data_min), abs(data_max))
cmin = -2
cmax = 2
colorscale = [
    (0.0, "rgb(255,0,0)"),   # Red
    (0.5, "rgb(200,200,200)"), # White
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

curves = []
freqplots = []
for freq_ind = [1,round(Int, 2*length(Ω)/3)]
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
                            ylabel="Re[G(v,ω)]",
                            xlabel="v [mV]",
                            margins=5Plots.mm,
                            size=(400,200)))
    Plots.savefig(string(figurepath,"conductance/freq_ind=",freq_ind,"_neuron_",n,".svg"))
end

azimuth = 140  # Azimuthal angle
elevation = 20  # Elevation angle
layout = Layout(
    autosize=false,
    title=string("Frequency-dependent conductance of HCO neuron ",n),
    scene = attr(
        xaxis = attr(title = "v [mV]"),
        yaxis = attr(title = "log₁₀(ω) [rad/ms]"),
        zaxis = attr(title = "Re[G(v,ω)]",
        range = [-4, 1]
        ),
        camera = attr(eye = spherical_eye(azimuth, elevation, 2.1)),
        scene_camera_eye=attr(x=5, y=0.88, z=-0.64),
        aspectratio = attr(x = 1, y = 1, z = 0.7),
    ),
    #width=500, 
    # height=500,
    margin=attr(l=0, r=0, b=20, t=0)  # Increased bottom margin to prevent cutting
)

plt = PlotlyJS.plot([pltf,curves...], layout)
PlotlyJS.savefig(plt,string(figurepath,"conductance/n=",n,"m=",m,".svg"))
plt
##

##
freqplots = []


plot(pltf)
pltf
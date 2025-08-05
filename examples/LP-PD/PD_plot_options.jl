expDict = Dict("Experiment 22/PD1"=>Dict(),
                "Experiment 23/PD1"=>Dict(),
                "Experiment 24/PD1"=>Dict(),
                "Experiment 24/PD2"=>Dict(),
                "Experiment 25/PD1"=>Dict())

expDict["Experiment 22/PD1"][:color] = palette(:default)[1]
expDict["Experiment 23/PD1"][:color] = palette(:default)[2]
expDict["Experiment 24/PD1"][:color] = palette(:default)[3]
expDict["Experiment 24/PD2"][:color] = palette(:default)[4]
expDict["Experiment 25/PD1"][:color] = palette(:default)[5]

expDict["Experiment 22/PD1"][:number] = 1
expDict["Experiment 23/PD1"][:number] = 3
expDict["Experiment 24/PD1"][:number] = 4
expDict["Experiment 24/PD2"][:number] = 5
expDict["Experiment 25/PD1"][:number] = 2

# Ornstein-Uhlenbeck process parameters
data = Dict("Experiment 22/PD1"=>(id="Experiment 22/973_130_00",numbers=9:12,marker=:star,opacity=1.0,color=palette(:default)[1],
                    number=Dict((α=0.015,σ=0.015) => 9, (α=0.01,σ=0.015) => 10, (α=0.005,σ=0.015) => 11, (α=0.001,σ=0.015) => 12,
                                (α=0.001,σ=0.03) => 13, (α=0.005,σ=0.03) => 14, (α=0.01,σ=0.03) => 15, (α=0.015,σ=0.03) => 16)),
                "Experiment 23/PD1"=>(id="Experiment 23/973_143_00",numbers=1:4,marker=:circle,opacity=1.0,color=palette(:default)[2],
                    number=Dict((α=0.015,σ=0.015) => 1, (α=0.01,σ=0.015) => 2, (α=0.005,σ=0.015) => 3, (α=0.001,σ=0.015) => 4,
                                (α=0.001,σ=0.03) => 12, (α=0.005,σ=0.03) => 11, (α=0.01,σ=0.03) => 10, (α=0.015,σ=0.03) => 9)),
                "Experiment 23/PD2"=>(id="Experiment 23/973_143_00",numbers=17:20,marker=:square,opacity=1.0,color=palette(:default)[3],
                    number=Dict((α=0.015,σ=0.015) => 17, (α=0.01,σ=0.015) => 18, (α=0.005,σ=0.015) => 19, (α=0.001,σ=0.015) => 20,
                                (α=0.001,σ=0.03) => 28, (α=0.005,σ=0.03) => 27, (α=0.01,σ=0.03) => 26, (α=0.015,σ=0.03) => 25)),
                "Experiment 24/PD1"=>(id="Experiment 24/973_143_1_00",numbers=1:4,marker=:diamond,opacity=1.0,color=palette(:default)[4],
                    number=Dict((α=0.015,σ=0.015) => 1, (α=0.01,σ=0.015) => 2, (α=0.005,σ=0.015) => 3, (α=0.001,σ=0.015) => 4,
                                (α=0.001,σ=0.03) => 12, (α=0.005,σ=0.03) => 11, (α=0.01,σ=0.03) => 10, (α=0.015,σ=0.03) => 9)),
                "Experiment 24/PD2"=>(id="Experiment 24/973_143_1_00",numbers=21:24,marker=:triangle,opacity=1.0,color=palette(:default)[5],
                    number=Dict((α=0.015,σ=0.015) => 21, (α=0.01,σ=0.015) => 22, (α=0.005,σ=0.015) => 23, (α=0.001,σ=0.015) => 24,
                                (α=0.001,σ=0.03) => 32, (α=0.005,σ=0.03) => 31, (α=0.01,σ=0.03) => 30, (α=0.015,σ=0.03) => 29)),
                "Experiment 25/PD1"=>(id="Experiment 25/973_146_00",numbers=21:24,marker=:utriangle,opacity=1.0,color=palette(:default)[6],
                    number=Dict((α=0.015,σ=0.015) => 1, (α=0.01,σ=0.015) => 2, (α=0.005,σ=0.015) => 3, (α=0.001,σ=0.015) => 4,
                                (α=0.001,σ=0.03) => 12, (α=0.005,σ=0.03) => 11, (α=0.01,σ=0.03) => 10, (α=0.015,σ=0.03) => 9)))
    
using Plots
k = 1   # =2 in Prinz et al. 2003

# Time constants

function tau_m_I_Na(V)
    τ =  1.32 - (1.26 / (1 + exp((V + 120) / -25.0)))
    return k*τ
end

function tau_h_I_Na(V)
    τ =  0.67 / (1 + exp((V + 62.9) / -10.0)) * (1.5 + 1 / (1 + exp((V + 34.9) / 3.6)))
    return k*τ
end

function tau_m_I_Kd(V)
    τ =  7.2 - (6.4 / (1 + exp((V + 28.3) / -19.2)))
    return k*τ
end

function tau_m_I_CaT(V)
    τ =  21.7 - (21.3 / (1 + exp((V + 68.1) / -20.5)))
    return k*τ
end

function tau_h_I_CaT(V)
    τ =  105 - (89.8 / (1 + exp((V + 55) / -16.9)))
    return k*τ
end

function tau_m_I_CaS(V)
    τ =  1.4 + 7/(exp((V + 27) / 10) + exp((V + 70) / -13))
    return k*τ
end

function tau_h_I_CaS(V)
    τ =  60 + 150/(exp((V + 55) / 9) + exp((V + 65) / -16))
    return k*τ
end

function tau_m_I_A(V)
    τ =  11.6 - (10.4 / (1 + exp((V + 32.9) / -15.2)))
    return k*τ
end

function tau_h_I_A(V)
    τ =  38.6 - (29.2 / (1 + exp((V + 38.9) / -26.5)))
    return k*τ
end

function tau_m_I_KCa(V)
    τ =  90.3 - (75.1 / (1 + exp((V + 46) / -22.7)))
    return k*τ
end

function tau_m_I_H(V)
    τ =  272 + 1499 / (1 + exp((V + 42.2) / -8.73))
    return k*τ
end

function tau_m_I_H_prinz(V)
    τ =  2/(exp((V+169.7)/-11.6)+exp((V-26.7)/14.3))
    return k*τ
end

function tau_Ca(V)
    τ =  20
    return k*τ
end

function tau_Ca_prinz(V)
    τ =  200
    return τ
end

# Activation functions 

# Function for I_Na m_\infty
function m_inf_I_Na(V)
    return 1 / (1 + exp((V + 25.5) / -5.29))
end

# Function for I_Na h_\infty
function h_inf_I_Na(V)
    return 1 / (1 + exp((V + 48.9) / 5.18))
end

# Function for I_CaT m_\infty
function m_inf_I_CaT(V)
    return 1 / (1 + exp((V + 27.1) / -7.2))
end

# Function for I_CaT h_\infty
function h_inf_I_CaT(V)
    return 1 / (1 + exp((V + 32.1) / 5.5))
end

# Function for I_CaS m_\infty
function m_inf_I_CaS(V)
    return 1 / (1 + exp((V + 33) / -8.1))
end

# Function for I_CaS h_\infty
function h_inf_I_CaS(V)
    return 1 / (1 + exp((V + 60) / 6.2))
end

# Function for I_A m_\infty
function m_inf_I_A(V)
    return 1 / (1 + exp((V + 27.2) / -8.7))
end

# Function for I_A h_\infty
function h_inf_I_A(V)
    return 1 / (1 + exp((V + 56.9) / 4.9))
end

# Function for I_KCa m_\infty
function m_inf_I_KCa_aux(V)
    return (1 / (1 + exp((V + 28.3) / -12.6)))
end
function m_inf_I_KCa(V, Ca)
    return Ca / (Ca + 3)*m_inf_I_KCa_aux(V)
end

# Function for I_Kd m_\infty
function m_inf_I_Kd(V)
    return 1 / (1 + exp((V + 12.3) / -11.8))
end

# Function for I_H m_\infty
function m_inf_I_H(V)
    return 1 / (1 + exp((V + 70) / 6))
end

function m_inf_I_H_prinz(V)
    return 1 / (1 + exp((V + 75) / 5.5))
end

function m_inf_Ca(ICA_T,ICA_S)
    return -0.94 * (ICA_T+ICA_S) + 0.05
end

function m_inf_Ca_prinz(ICA_T,ICA_S)
    return -14.96 * ICA_T+ICA_S + 0.05
end

## Plot tau functions

V = -70:0.01:-20
plt0 = plot(V, tau_m_I_Na.(V), label="I_Na τ_m", xlabel="V (mV)", ylabel="τ (ms)",title="Fast",color=:red)
plt0 = plot!(V, tau_h_I_Na.(V), label="I_Na τ_h",color=:red)
plt0 = plot!(ylim=(0, 1.5))#,title="Na is excitatory"

plt1=plot(V, tau_m_I_Kd.(V), label="I_Kd τ_m", xlabel="V (mV)", ylabel="τ (ms)",color=:blue)
plt1=plot!(V, tau_m_I_CaT.(V), label="I_CaT τ_m", xlabel="V (mV)", ylabel="τ (ms)",color=:cyan)
plt1=plot!(V, tau_m_I_A.(V), label="I_A τ_m", xlabel="V (mV)", ylabel="τ (ms)",color=:green)
plt1=plot!(ylim=(0, 12))#,title="K and A are inhibitory, Ca is excitatory"

plt2=plot(V, tau_h_I_CaT.(V), label="I_CaT τ_h",color=:cyan)
plt2=plot!(V, tau_h_I_A.(V), label="I_A τ_h",color=:green)
plt2=plot!(V, tau_m_I_CaS.(V), label="I_CaS τ_m", xlabel="V (mV)", ylabel="τ (ms)",color=:orange)
plt2=plot!(V, tau_m_I_KCa.(V), label="I_KCa τ_m", xlabel="V (mV)", ylabel="τ (ms)",color=:magenta,ylim=(0, maximum(tau_m_I_KCa.(V))))
plt2=plot!(V, tau_Ca.(V), label="Ca τ_m",color=:black)
plt2=plot!(ylim=(0, 80))#,title="Ca is excitatory, A and KCa are inhibitory"

plt3=plot(V, tau_h_I_CaS.(V), label="I_CaS τ_h", xlabel="V (mV)", ylabel="τ (ms)",color=:orange)
plt3=plot!(V, tau_m_I_H.(V), label="I_H τ_m",color=:purple)
plt3=plot!(V, tau_m_I_H_prinz.(V), label="I_H τ_m (Prinz)",color=:purple)
plt3=plot!(V, tau_Ca_prinz.(V), label="Ca τ_m (Prinz)",color=:black)
plt3=plot!(ylim=(0, 700))#,title="Ca, H is inhibitory"

plot(plt0, plt1, plt2, plt3, layout=(2,2), size=(1000, 800),legend=true)
# savefig("tau_functions.png")

## IV curves
ENa = 50
EK = -80
EA = EK
ECa = 130
EH = -20
EL = -50


IV_Na(V) = 100*m_inf_I_Na(V)^3*h_inf_I_Na(V)*(V-ENa)
IV_Kd(V) = 100*m_inf_I_Kd(V)^4*(V-EK)
IV_CaT(V) = 2.5*m_inf_I_CaT(V)^3*h_inf_I_CaT(V)*(V-ECa)
IV_CaS(V) = 6*m_inf_I_CaS(V)^3*h_inf_I_CaS(V)*(V-ECa)
IV_A(V) = 50*m_inf_I_A(V)^3*h_inf_I_A(V)*(V-EA)
IV_KCa(V) = 5*m_inf_I_KCa(V,m_inf_Ca(IV_CaT(V),IV_CaS(V)))^4*(V-EK)
IV_H(V) = 0.01*m_inf_I_H(V)*(V-EH)
IV_L(V) = 0.01*(V-EL)

IV(V) = [IV_Na.(V),IV_Kd.(V),IV_CaT.(V),IV_CaS.(V),IV_A.(V),IV_KCa.(V),IV_H.(V),IV_L.(V)]

V = -70:0.01:-0
plt = [plot(V,IV(V)[i],label=["I_Na" "I_Kd" "I_CaT" "I_CaS" "I_A" "I_KCa" "I_H" "I_L"][i],xlabel="V (mV)",ylabel="I (nA)",title="IV curves") for i=1:8]
plot(plt...,layout=(4,2),size=(1000,800),legend=true)


## Plot activation functions 

V = collect(-70:0.01:0.0)
plt0 = plot(V, m_inf_I_Na.(V), label="I_Na m∞", xlabel="V (mV)", ylabel="τ (ms)",title="Fast")
plt0 = plot!(V, h_inf_I_Na.(V), label="I_Na h∞")
plt0 = plot!(ylim=(0, 1.0),title="Na is excitatory")

plt1=plot(V, m_inf_I_Kd.(V), label="I_Kd m∞", xlabel="V (mV)", ylabel="τ (ms)")
plt1=plot!(V, m_inf_I_CaT.(V), label="I_CaT m∞", xlabel="V (mV)", ylabel="τ (ms)")
plt1=plot!(V, m_inf_I_A.(V), label="I_A m∞", xlabel="V (mV)", ylabel="τ (ms)")
plt1=plot!(ylim=(0, 1.0),title="K and A are inhibitory, Ca is excitatory")

plt2=plot(V, h_inf_I_CaT.(V), label="I_CaT h∞")
plt2=plot!(V, h_inf_I_A.(V), label="I_A h∞")
plt2=plot!(V, m_inf_I_KCa_aux.(V), label="I_KCa m∞-aux")
plt2=plot!(V, m_inf_I_CaS.(V), label="I_CaS m∞", xlabel="V (mV)", ylabel="τ (ms)")
CA = m_inf_Ca.(IV_CaT.(V),IV_CaS.(V))
plt2=plot!(V, m_inf_I_KCa.(V,CA), label="I_KCa m∞", xlabel="V (mV)", ylabel="τ (ms)",ylim=(0, maximum(m_inf_I_KCa.(V,CA))))
plt2=plot!(ylim=(0, 1.0),title="Ca is excitatory, A and KCa are inhibitory")

plt3=plot(V, h_inf_I_CaS.(V), label="I_CaS h∞", xlabel="V (mV)", ylabel="τ (ms)")
plt3=plot!(V, m_inf_I_H.(V), label="I_H m∞")
plt3=plot!(V, m_inf_I_H_prinz.(V), label="I_H m∞ (Prinz)")
plt3=plot!(ylim=(0, 1.0),title="Ca is excitatory, H is inhibitory")

plot(plt0, plt1, plt2, plt3, layout=(2,2), size=(1000, 800),legend=true)
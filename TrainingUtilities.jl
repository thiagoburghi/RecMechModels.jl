"""
    Main training function
        ⋅ loss is the loss function used for training 
        ⋅ net is the full recurrent biological neural network model
        ⋅ ind_train is a vector containing the indices of the neurons in the network that will be trained. 
        ⋅ data is the training data, a vector of DataLoader elements
        ⋅ opt is the optimization algorithm
        ⋅ clvaldata is the data used to log closed-loop performance. if empty, closed-loop performance is not logged.
"""
function train_ANN(loss, net::Network, trainData, valdata, opt, xpu; epochs=1000, print_interval=1, snapshots_interval = 10)
    local minibatch_loss
    local cl_log_counter
    train_time = zeros(epochs)
    train_loss = zeros(epochs)
    val_loss = zeros(epochs)
    
    net = xpu(net)
    loss = xpu(loss)
    
    # Initialize an array to hold the last models                
    model_snapshots = []
    cl_log_counter = 1
    cumtime = 0
    
    # For old Flux training syntax:
    # θ = Flux.params(net)
    
    # For new Flux training syntax:
    opt_state = Flux.setup(opt,net)
    # Flux.adjust!(opt_state.cell.Cinv, 0.1)
    # Flux.adjust!(opt_state.cell.Cinv, beta=(0.8,0.9))
    # Flux.adjust!(opt_state.cell.ANN[1,1][:leakCurrent], 0.1)
    # Flux.adjust!(opt_state.cell.ANN[1,1][:leakCurrent], beta=(0.8,0.9))

    for i = 1:epochs
        total_loss = 0
        epoch_time = @elapsed for d in trainData
            # Old Flux training syntax:
            # gs = gradient(θ) do
            #     minibatch_loss = loss(d...,net.cell)
            #     minibatch_loss += regularize(loss,net.cell)
            #     return minibatch_loss
            # end

            # New Flux training syntax:
            gs = Flux.gradient(net) do m
                minibatch_loss = loss(d...,m.cell)
                minibatch_loss += regularize(loss,m.cell)
                return minibatch_loss
            end
            total_loss += minibatch_loss

            # Old Flux training syntax:
            # Flux.update!(opt, θ, gs)

            # New Flux training syntax:
            Flux.update!(opt_state, net, gs[1])
        end
        cumtime += epoch_time
        train_loss[i] = total_loss/length(trainData)
        train_time[i] = cumtime

        # Open-loop validation loss
        total_loss = 0
        for d in valdata
            total_loss += loss(d...,net.cell)
        end
        val_loss[i] = total_loss/length(valdata)

        if (i % print_interval == 0) || i == epochs
        println("Epoch ", i, ": open-loop training loss = ", train_loss[i], 
                                "; validation loss = ",val_loss[i], 
                                "; time = ",floor(Int,cumtime/60),"m ",floor(Int,rem(cumtime,60)),"s ")
        end
        
        if (i % snapshots_interval == 0) || i == epochs
            push!(model_snapshots, (model=deepcopy(cpu(net)), epoch=i, loss=train_loss[i], time=train_time[i]))
        end
    end
    println("Total training time: ",floor(Int,cumtime/60),"m ",floor(Int,rem(cumtime,60)),"s ")
    return (net=cpu(net),opt=cpu(opt),time=train_time,trainLoss=train_loss,valLoss=val_loss,snapshots=model_snapshots)
end

"""
    Closed-loop fitting
"""
function trainEpoch!(net::N,lossFun::L,trainData::D,opt::O,θ::P; slow_ic=10) where {N<:Network,L<:AbstractLoss,D<:AbstractData,O<:Flux.Optimise.AbstractOptimiser,P<:Flux.Params}
    loss = 0
    gs = gradient(θ) do
        loss = lossFun(net,trainData)
    end
    # Flux.update!(opt, θ, gs)
    Flux.update!(opt, Flux.params(net), gs)
    opt.eta = opt.eta/slow_ic
    Flux.update!(opt, Flux.params(trainData), gs)
    opt.eta = opt.eta*slow_ic
    return loss
end

function trainEpoch!(net::N,lossFun::L,trainData::MiniBatches,opt::O,θ::P; slow_ic=10) where {N<:Network,L<:AbstractLoss,O<:Flux.Optimise.AbstractOptimiser,P<:Flux.Params}
    local minibatch_loss
    total_loss = 0
    for d in trainData
        gs = gradient(θ) do
            minibatch_loss = lossFun(net,d)
            return minibatch_loss
        end
        total_loss += minibatch_loss
        # Flux.update!(opt, θ, gs)
        Flux.update!(opt, Flux.params(net), gs)
        opt.eta = opt.eta/slow_ic
        Flux.update!(opt, Flux.params(trainData), gs)
        opt.eta = opt.eta*slow_ic
    end
    return total_loss/length(trainData)
end

function train(net::N,lossFun::L,trainData::D,opt; xpu=gpu, epochs=100, snapshots_interval = 10, data_snapshots=false, print_interval=100, slow_ic=10, opt_schedule::Union{Nothing,Tuple}=nothing, teacher_schedule::Union{Nothing,Vector}=nothing) where {N<:Network,L<:AbstractLoss,D<:AbstractData}
    cumtime = 0
    train_loss = zeros(epochs)
    train_time = zeros(epochs)
    model_snapshots = []
    teacher_counter = 1
    opt_counter = 1

    net,trainData,lossFun = map(xpu,(net,trainData,lossFun))
    θ = Flux.params(net,trainData)

    # Keep the first model for comparison
    initial_loss = lossFun(net,trainData)
    println("Epoch ", 0, ": training loss = ", initial_loss)
    push!(model_snapshots, (model=cpu(deepcopy(net)), epoch=0, loss=initial_loss, time=0.0, data=(data_snapshots ? deepcopy(cpu(trainData)) : nothing)))

    for i = 1:epochs
        # Teacher forcing schedule
        if !isnothing(teacher_schedule)
            epoch = teacher_schedule[teacher_counter][1]
            if i == epoch
                γ = teacher_schedule[teacher_counter][2]
                println("Changing γ to ",γ,", scheduled element: ",teacher_counter)
                net.cell.γ[1] = gpu([γ;;])
                net.cell.γ[2] = gpu([γ;;])
                if teacher_counter == length(teacher_schedule)
                    println("End of generalized teacher forcing schedule.")
                    teacher_schedule = nothing
                else
                    teacher_counter += 1 
                end
            end
        end
        # Optimizer schedule
        if !isnothing(opt_schedule)
            if i == opt_schedule[opt_counter][:epoch]
                println("Changing η from ",opt.eta," to ",opt_schedule[opt_counter][:η])
                opt.eta = opt_schedule[opt_counter][:η]
                if opt_counter == length(opt_schedule)
                    println("End of optimizer schedule.")
                    opt_schedule = nothing
                else
                    opt_counter += 1 
                end
            end
        end
        # Train one epoch
        epoch_time = @elapsed begin
            train_loss[i] = trainEpoch!(net,lossFun,trainData,opt,θ; slow_ic=slow_ic)
        end
        cumtime += epoch_time
        train_time[i] = cumtime

        # Print training loss every given number of epochs
        if (i % print_interval == 0) || i == epochs
            println("Epoch ", i, ": training loss = ", train_loss[i], 
                "; time = ",floor(Int,cumtime/60),"m ",floor(Int,rem(cumtime,60)),"s ")
        end

        # Save models every given number of epochs
        if (i % snapshots_interval == 0) || i == epochs
            model = deepcopy(cpu(net))
            # Reduce model size for saving
            model.V = map(v->v[:,1:1],model.V)
            model.X = map(X->map(x->x[:,1:1],X),model.X)
            push!(model_snapshots, (model=model, epoch=i, loss=train_loss[i], time=train_time[i],data=(data_snapshots ? deepcopy(cpu(trainData)) : nothing)))
        end
    end
    return (net=cpu(net),data=cpu(trainData),time=train_time,loss=train_loss,snapshots=model_snapshots)
end
module Kalman
using Parameters

struct State
    x::Vector{Float64}
    P::Matrix{Float64}
end

#interface
abstract type Process end
abstract type Observer end


mutable struct KalmanFilter
    state::State
    process::Process
    observer::Observer
end

state(kalman::KalmanFilter) = kalman.state.x
Base.getindex(kalman::KalmanFilter,idx::Int) = kalman.state.x[idx]

function predict(kalman::KalmanFilter;steps::Int = 1)
    state = kalman.state
    for i=1:steps
        state = predict(kalman.process,state)
    end
    state
end

function predict!(kalman::KalmanFilter;steps::Int = 1)
    kalman.state = predict(kalman; steps = steps)
end

function update!(kalman::KalmanFilter,measurement; observer::Observer = kalman.observer)
    kalman.state = update!(observer,kalman.state,measurement)
end



#interface general functions
function predict(process::Process,state::State)
    State(process(state),err_cov(process,state))
end

function update!(observer::Observer,state::State,measurement)
    innovation = measurement - observer(state)

    P_new,kalman_gain = err_cov(observer,state)

    x_new = state.x + kalman_gain * innovation

    state_new = State(x_new,P_new)

    residual = measurement - observer(state_new)

    @pack observer = innovation,kalman_gain,residual
    state_new
end


include("linear.jl")
include("nonlinear.jl")
export predict,predict!,KalmanFilter,update!,state
end

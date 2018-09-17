#linear
using ForwardDiff

struct NonLinearProcess <: Process
    F::Function
    Q::Matrix{Float64}
end
(process::NonLinearProcess)(state::State) = begin
    process.F(state.x)
end
Process(F::Function,Q::Matrix) = NonLinearProcess(F,Q)

err_cov(process::NonLinearProcess,state::State) = begin
    @unpack F,Q = process
    Fjac = ForwardDiff.jacobian(F,state.x)
    err_cov(LinearProcess(Fjac,Q),state)
    # P = state.P
    # F*P*F' + Q
end


@with_kw_noshow mutable struct NonLinearObserver <: Observer
    H::Function
    R::Matrix

    kalman_gain::Matrix = zeros(1,1)
    innovation::Vector = zeros(1)
    residual::Vector = zeros(1)
end
(observer::NonLinearObserver)(state::State) = begin
    observer.H(state.x)
end
Observer(H::Function,R::Matrix) = NonLinearObserver(H = H,R = R)

err_cov(observer::NonLinearObserver,state::State) = begin
    @unpack R,H = observer
    @unpack x,P = state
    Hjac = ForwardDiff.jacobian(H,x)
    err_cov(LinearObserver(H=Hjac,R=R))
    # S = H*P*H' + R
    # K = P * H' * inv(S)
    #
    # temp = (I-K*H)
    # P_new = temp * P * temp' + K*R*K'
    #
    # P_new,K
end

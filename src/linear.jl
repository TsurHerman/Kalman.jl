#linear

struct LinearProcess <: Process
    F::Matrix{Float64}
    Q::Matrix{Float64}
end
(process::LinearProcess)(state::State) = begin
    process.F * state.x
end
Process(F::Matrix,Q::Matrix) = LinearProcess(F,Q)

err_cov(process::LinearProcess,state::State) = begin

    @unpack F,Q = process
    # Q .= 0.95*Q
    P = state.P
    F*P*F' + Q
end


@with_kw_noshow mutable struct LinearObserver <: Observer
    H::Matrix
    R::Matrix

    kalman_gain::Matrix = zeros(1,1)
    innovation::Vector = zeros(1)
    residual::Vector = zeros(1)
end
(observer::LinearObserver)(state::State) = begin
    observer.H * state.x
end
Observer(H::Matrix,R::Matrix) = LinearObserver(H = H,R = R)

err_cov(observer::LinearObserver,state::State) = begin
    @unpack R,H = observer
    @unpack x,P = state
    S = H*P*H' + R
    K = P * H' * inv(S)

    temp = (I-K*H)
    P_new = temp * P * temp' + K*R*K'

    P_new,K
end

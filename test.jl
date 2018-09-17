# using Revise
# using Kalman
# using PyPlot
# PyPlot.ion()

using Kalman
A = (0:1:800) + 20
Ad = A + 25*(rand(length(A)) - 0.5)

plot(A,"b")
plot(Ad,"r-")

x0 = [0.0 ,0.0 ,1.0]
P0 = 00ones(3,3)

state0 = Kalman.State(x0,P0)

std_process = 0.001*[1 , 0.1,3]

F(x) = [x[1] + x[2] + x[3]^2,x[2] + x[3],x[3] * 0.8]

Q = std_process*std_process'

process = Kalman.Process(F,Q)

H = [1.0 0.0 0.0]
R = ones(1,1)*52
observer = Kalman.Observer(H,R)

kalman = KalmanFilter(state0,process,observer)

filt = Float64[]e
dif = Float64[]

@time for i=1:length(Ad)
    update!(kalman,[Ad[i]])

    push!(filt,predict(kalman;steps = 3).x[1])
    push!(dif,kalman.observer.residual[1])

    predict!(kalman)

    # push!(filt,kalman.state.x[1])
end
unshift!(filt,0)
unshift!(filt,0)
unshift!(filt,0)

plot(filt)
plot(dif)
kalman
dif

mean(dif)
# dif

predict(kalman;steps = 20)

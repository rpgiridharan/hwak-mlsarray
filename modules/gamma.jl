using LinearAlgebra

function gam_p(ky, kap, C)
    om_p = zeros(ComplexF64, size(ky))
    kysq = ky.^2
    common = -im*C*(1 .+ kysq)./(2 .* kysq)
    extra = sqrt.(- C^2 .* (1 .+ 1 ./ kysq).^2 .+ 4im*C .* ky .* kap ./ kysq) ./ 2
    om_p = common .+ extra
    return imag.(om_p)
end

function gammax(ky, kap, C)
    return maximum(gam_p(ky, kap, C))
end

function kymax(ky, kap, C)
    values = gam_p(ky, kap, C)
    _, index = findmax(values)
    return ky[index]
end
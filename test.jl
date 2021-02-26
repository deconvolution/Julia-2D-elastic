@parallel_indices (ixin,izin) function x_2_end(in,out)
out[ixin,izin]=in[ixin-1,izin];
return nothing
end

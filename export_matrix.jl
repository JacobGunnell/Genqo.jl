using Printf

function export_matrix(matrix, file_path)
    open(file_path, "w") do file
        for i = 1:size(matrix, 1)
            row = matrix[i, :]
            write(file, replace(join((@sprintf("%.8e%c%.8ej", real(x), if sign(imag(x)) == -1 "-" else "+" end, abs(imag(x))) for x in row), ", "), "im" => "j") * "\n")
        end
    end
end
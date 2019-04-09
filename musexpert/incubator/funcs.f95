module funcs
    use iso_c_binding
    implicit none

    interface quickselect
        module procedure quickselect_real, quickselect_integer
    end interface quickselect
    interface swap
    	module procedure swap_integer, swap_real
    end interface swap

contains
    integer function quickselect_integer(A, k)
        integer A(:)
        integer :: s, e, i, j, k

        s = 1
        e = size(A)
        do while(s .ne. e)
            call swap(A(k), A(e))
            j = s
            do i=s, e-1
                if (A(i)<A(e)) then
                    call swap(A(i), A(j))
                    j = j+1
                end if
            end do
            call swap(A(j), A(e))
            if (j==k) then
                quickselect_integer = A(k)
                exit
            else if (j<k) then
                s = j+1
            else
                e = j-1
            end if
        end do
        quickselect_integer = A(k)
    end function quickselect_integer
    real function quickselect_real(A, k)
        real(kind=C_FLOAT) :: A(:)
        integer(kind=C_INTPTR_T) :: k
        integer(kind=C_INTPTR_T) :: s, e, i, j

        s = 1
        e = size(A)
        do while(s .ne. e)
            call swap(A(k), A(e))
            j = s
            do i=s, e-1
                if (A(i)<A(e)) then
                    call swap(A(i), A(j))
                    j = j+1
                end if
            end do
            call swap(A(j), A(e))
            if (j==k) then
                exit
            else if (j<k) then
                s = j+1
            else
                e = j-1
            end if
        end do
        quickselect_real = A(k)
    end function quickselect_real

    subroutine swap_integer(a, b)
        integer, intent(inout) :: a, b
        integer c
        c = a
        a = b
        b = c
    end subroutine swap_integer
    subroutine swap_real(a, b)
        real, intent(inout) :: a, b
        real c
        c = a
        a = b
        b = c
    end subroutine swap_real

end module funcs

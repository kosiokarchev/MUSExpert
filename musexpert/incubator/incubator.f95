module incubator_mod
    use iso_c_binding
    use ieee_arithmetic
    use funcs
    implicit none

    type Cube
        integer(kind=C_INTPTR_T), pointer :: naxes(:)
        integer(kind=C_INTPTR_T) :: rgbsplit(2)

        real(kind=C_FLOAT), pointer :: d(:,:,:)
        real(kind=C_FLOAT), pointer :: w(:,:,:)
        real(kind=C_FLOAT), pointer :: white(:,:)
        real(kind=C_FLOAT), pointer :: red(:,:)
        real(kind=C_FLOAT), pointer :: green(:,:)
        real(kind=C_FLOAT), pointer :: blue(:,:)
        real(kind=C_FLOAT), pointer :: inv_var(:,:)
    end type Cube

    abstract interface
        subroutine callback(callback_cube, imslice, i)
            use iso_c_binding
            type(C_PTR), value :: callback_cube
            real(kind=C_FLOAT), intent(in) :: imslice
            integer(kind=C_INTPTR_T), intent(in) :: i
        end subroutine callback
    end interface

contains
    type(C_PTR) function cube_init(naxes_ptr, data_ptr, var_ptr) bind(c)
        type(C_PTR), value :: naxes_ptr
        type(C_PTR), value :: data_ptr
        type(C_PTR), value :: var_ptr

        type(cube), pointer :: c

        allocate(c)
        call c_f_pointer(naxes_ptr, c%naxes, shape=[3])
        call c_f_pointer(data_ptr, c%d, shape=c%naxes)
        call c_f_pointer(var_ptr, c%w, shape=c%naxes)

        c%rgbsplit(1) = c%naxes(3)/3
        c%rgbsplit(2) = 2*c%rgbsplit(1)

        cube_init = c_loc(c)

    end function cube_init

    subroutine cube_flats(cube_ptr) bind(c)
        type(C_PTR), value :: cube_ptr
        type(cube), pointer :: c

        real(kind=C_FLOAT), allocatable :: res_cube(:,:,:)

        call c_f_pointer(cube_ptr, c)

        allocate(c%white(c%naxes(1), c%naxes(2)))
        allocate(c%red(c%naxes(1), c%naxes(2)))
        allocate(c%green(c%naxes(1), c%naxes(2)))
        allocate(c%blue(c%naxes(1), c%naxes(2)))
        allocate(c%inv_var(c%naxes(1), c%naxes(2)))

        !$acc data copyin(c%d, c%w, c%naxes) copyout(c%white, c%red(:,:), c%green(:,:), c%blue(:,:), c%inv_var(:,:)) create(res_cube(:, :, :))

        !!$acc parallel
        !allocate(res_cube(c%naxes(1), c%naxes(2), c%naxes(3)))
        !!$acc end parallel

        !$acc kernels

        where(isnan(c%w))
            c%w = ieee_value(c%w, ieee_positive_inf)    ! Clean variance
        end where

        print *, "Creating inverse variance image..."
        c%inv_var = c%naxes(3) / sum(c%w, dim=3)        ! inv_variance = N / sum(variance)

        print *, "Creating weights cube..."
        c%w = 1.0 / c%w                                 ! var -> weight

        res_cube = c%d * c%w                            ! res_cube -> weighted data

        print *, "Creating R, G, B..."
        c%red = sum(res_cube(:,:, : c%rgbsplit(1)), dim=3)
        c%green = sum(res_cube(:,:, c%rgbsplit(1)+1 : c%rgbsplit(2)), dim=3)
        c%blue = sum(res_cube(:,:, c%rgbsplit(2)+1 :), dim=3)

        !!$acc parallel
        !deallocate(res_cube)
        !!$acc end parallel

        print *, "Creating white image..."
        c%white = (c%red + c%green + c%blue)

        print *, "Normalizing R, G, B..."
        c%red = c%red / sum(c%w(:,:, : c%rgbsplit(1)), dim=3)
        c%green = c%green / sum(c%w(:,:, c%rgbsplit(1)+1 : c%rgbsplit(2)), dim=3)
        c%blue = c%blue / sum(c%w(:,:, c%rgbsplit(2)+1 :), dim=3)

        print *, "Normalizing white image..."
        c%white = c%white / sum(c%w, dim=3)

    end subroutine cube_flats

    subroutine cube_get_flats(cube_ptr, white, red, green, blue, inv_var) bind(c)
        type(C_PTR), value :: cube_ptr
        type(cube), pointer :: c

        type(C_PTR), intent(out) :: white, red, green, blue, inv_var

        call c_f_pointer(cube_ptr, c)

        white = c_loc(c%white)
        red = c_loc(c%red)
        green = c_loc(c%green)
        blue = c_loc(c%blue)
        inv_var = c_loc(c%inv_var)

    end subroutine cube_get_flats

    subroutine cube_deallocate(cube_ptr) bind(c)
        type(C_PTR), value :: cube_ptr
        type(cube), pointer :: c
        call c_f_pointer(cube_ptr, c)

        deallocate(c%white, c%red, c%green, c%blue, c%inv_var)
    end subroutine cube_deallocate

    subroutine cube_print(cube_ptr) bind(c)
        type(C_PTR), value :: cube_ptr
        type(cube), pointer :: c
        call c_f_pointer(cube_ptr, c)

        print *, "A cube: ", c%naxes
!        print *, c%d
!        print *, c%w
!        print *, c%inv_var
!        print *, c%red
!        print *, c%green
!        print *, c%blue
!        print *, c%white

    end subroutine cube_print

    subroutine muselet_step1(cube_ptr, left, cw, lwsize, lw, callback_ptr, callback_cube) bind(c)
        type(C_PTR), value :: cube_ptr
        type(cube), pointer :: c

        integer(kind=C_INTPTR_T), value :: left, cw
        integer(kind=C_INTPTR_T), value :: lwsize
        real(kind=C_DOUBLE), intent(in) :: lw(lwsize)

        type(C_FUNPTR), value :: callback_ptr
        procedure(callback), pointer :: cb
        type(C_PTR), value :: callback_cube

        integer(kind=C_INTPTR_T) leftmax, leftmin, rightmax, rightmin
        real(kind=C_FLOAT), allocatable :: contleft(:,:)
        real(kind=C_FLOAT), allocatable :: contright(:,:)
        real(kind=C_FLOAT), allocatable :: imslice(:,:)

        integer(kind=C_INTPTR_T) i, count


        call c_f_pointer(cube_ptr, c)
        call c_f_procpointer(callback_ptr, cb)

        count = 0

        !$ print *, "Running in parallel"
        !$OMP PARALLEL PRIVATE(leftmax, leftmin, rightmax, rightmin, contleft, contright, imslice)
        allocate(contleft(c%naxes(1), c%naxes(2)))
        allocate(contright(c%naxes(1), c%naxes(2)))
        allocate(imslice(c%naxes(1), c%naxes(2)))
        !$OMP DO
        do i=left+1, c%naxes(3) - (lwsize - left) -1
            leftmax = i-left
            leftmin = max(1, leftmax-cw+1)
            rightmin = leftmax+1 + lwsize
            rightmax = min(c%naxes(3), rightmin+cw-1)

            call get_imslice(c, leftmax+1, lw, imslice)
            call get_cont(c, leftmin, leftmax, contleft)
            call get_cont(c, rightmin, rightmax, contright)

            contleft = ((leftmax+1-leftmin)*contleft&
                    +(rightmax+1-rightmin)*contright)&
                    /(leftmax-leftmin + rightmax-rightmin + 2)

            imslice = imslice - contleft

!            write (filename, "(a,'nb/nb',i0.4,'.fits')") trim(m%workdir), i

!            call muselet_write_img(m, filename, imslice, 42+omp_get_thread_num())
            !            call fits_write(filename, imslice_ptr, 2, shape(imslice), 42+omp_get_thread_num())
            !            call fits_write(filename, imslice_ptr, 2, shape(imslice))

            !$OMP ATOMIC
            count = count + 1
            call cb(callback_cube, imslice(1, 1), i)
!            write (*,  "(i5,'/',i0)") count, c%naxes(3)-lwsize-1
!            print *, count
!            write (*, "('\r',i5,'/',i0)", advance="no") count, c%naxes(3)-lwsize-1
        end do
        !$OMP END DO
        !$OMP END PARALLEL
        print *, "Muselet step 1 complete."
    end subroutine muselet_step1

    subroutine get_imslice(c, k, lw, imslice)
        type (Cube), intent(in) :: c
        integer(kind=C_INTPTR_T), intent(in) :: k
        real(kind=C_DOUBLE), intent(in) :: lw(:)
        real(kind=C_FLOAT), intent(out) :: imslice(:,:)

        integer(kind=C_INTPTR_T) :: i, j, k1
        real(kind=C_FLOAT) :: weights(size(lw))

        k1 = k+size(lw)-1

        do j=1, c%naxes(2)
            do i=1, c%naxes(1)
                weights = lw * c%w(i, j, k:k1)
                imslice(i, j) = dot_product(c%d(i, j, k:k1), weights) / sum(weights)
            end do
        end do

    end subroutine get_imslice

    subroutine get_cont(c, left, right, cont)
        type (Cube), intent(in) :: c
        integer(kind=C_INTPTR_T), intent(in) :: left
        integer(kind=C_INTPTR_T), intent(in) :: right
        real(kind=C_FLOAT), intent(out) :: cont(:,:)

        real(kind=C_FLOAT) :: sortable(left:right)
        integer(kind=C_INTPTR_T) :: i, j, nplanes
        nplanes = right+1-left

        if (nplanes .eq. 1) then
            cont = c%d(:, :, left)
        else
            do j=1, c%naxes(2)
                do i=1, c%naxes(1)
                    sortable = c%d(i, j, left:right)
                    cont(i, j) = quickselect(sortable, nplanes/2)
                end do
            end do
        end if

    end subroutine get_cont

end module incubator_mod
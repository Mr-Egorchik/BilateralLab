# BilateralLab

Параллелизм достигается благодаря независимости вычисления новых значений пикселей, т.е. каждая нить независимо от остальных будет проводить вычисления для своего пикселя.

Привяжем текстуру к массиву, который хранит исходное изображение. Далее каждая нить будет извлекать из него 9 нужных пикселей и находить значение в соответствующем пикселе. Функции g и r реализованы отдельно с модификатором __device__ (вызывается из девайса и выполняется на девайсе).

Далее приведем результаты для изображений разных размеров

> Image size:212x252\
> CPU timme: 0.020745 sec.\
> Malloc, memcpy, bind texture: 0.002012 sec.\
> Filtering: 0.001080 sec.\
> Memcpy: 0.000786 sec.\
> Total GPU time: 0.003878 sec.\
> Speedup with total time: 5.349385\
> Speedup with only calc: 19.207764

Видим значительное ускорение в вычислениях. Отметим, что выделение памяти и привязка текстуры занимает почти вдвое больше времени, чем сами вычисления.

Теперь рассмотрим более крупное изображение

> Image size:3258x2500\
> CPU timme: 3.773754 sec.\
> Malloc, memcpy, bind texture: 0.021573 sec.\
> Filtering: 0.147997 sec.\
> Memcpy: 0.020021 sec.\
> Total GPU time: 0.189591 sec.\
> Speedup with total time: 19.904665\
> Speedup with only calc: 25.498834

Видим, что основной вклад во время вычислений теперь вносит непосредственно вычисление нового изображения, а ускорение стало больше, чем в случае маленькой картинки.
Возьмем картинку еще больше.

>Image size:4160x3000\
>CPU timme: 6.709158 sec.\
>Malloc, memcpy, bind texture: 0.017834 sec.\
>Filtering: 0.207382 sec.\
>Memcpy: 0.021545 sec.\
>Total GPU time: 0.246761 sec.\
>Speedup with total time: 27.188896\
>Speedup with only calc: 32.351731

Как видим, усоркение продолжает расти.

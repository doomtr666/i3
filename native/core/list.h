#pragma once

// macro for double linked lists
// works with any structure having a next and a prev field
// but you can use i3_dlist_elem as a convienience to declare them.
// You can also use i3_dlist to define a list head.

#define i3_dlist_elem(type) \
    struct                  \
    {                       \
        type *prev;         \
        type *next;         \
    }

#define i3_dlist(type) \
    struct             \
    {                  \
        type *first;   \
        type *last;    \
    }

#define i3_dlist_init(head) ((head)->first = (head)->last = NULL)
#define i3_dlist_copy(dst, src) ((dst)->first = (src)->first, (dst)->last = (src)->last)

#define i3_dlist_first(head) ((head)->first)
#define i3_dlist_last(head) ((head)->last)
#define i3_dlist_next(elem) ((elem)->next)
#define i3_dlist_prev(elem) ((elem)->prev)

#define i3_dlist_empty(head) (i3_dlist_first(head) == NULL)

#define i3_dlist_append(head, elem)              \
    do                                           \
    {                                            \
        assert((head) != NULL);                  \
        assert((elem) != NULL);                  \
                                                 \
        if ((head)->first == NULL)               \
        {                                        \
            (elem)->prev = (elem)->next = NULL;  \
            (head)->first = (head)->last = elem; \
        }                                        \
        else                                     \
        {                                        \
            (elem)->prev = (head)->last;         \
            (elem)->next = NULL;                 \
            (head)->last->next = (elem);         \
            (head)->last = (elem);               \
        }                                        \
    } while (0)

#define i3_dlist_prepend(head, elem)             \
    do                                           \
    {                                            \
        assert((head) != NULL);                  \
        assert((elem) != NULL);                  \
                                                 \
        if ((head)->first == NULL)               \
        {                                        \
            (elem)->prev = (elem)->next = NULL;  \
            (head)->first = (head)->last = elem; \
        }                                        \
        else                                     \
        {                                        \
            (elem)->next = (head)->first;        \
            (elem)->prev = NULL;                 \
            (head)->first->prev = (elem);        \
            (head)->first = (elem);              \
        }                                        \
    } while (0)

#define i3_dlist_remove(head, elem)            \
    do                                         \
    {                                          \
        assert((head) != NULL);                \
        assert((elem) != NULL);                \
                                               \
        if ((elem)->prev != NULL)              \
            (elem)->prev->next = (elem)->next; \
        else                                   \
            (head)->first = (elem)->next;      \
        if ((elem)->next != NULL)              \
            (elem)->next->prev = (elem)->prev; \
        else                                   \
            (head)->last = (elem)->prev;       \
    } while (0)

#define i3_dlist_foreach(head, it) for ((it) = ((head)->first); it; (it) = (it)->next)
#define i3_dlist_foreach_r(head, it) for ((it) = ((head)->last); it; (it) = (it)->prev)

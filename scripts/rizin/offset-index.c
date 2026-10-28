/* Regression for address-index removal: all spaces, aliases, neighbours,
 * an absent address, and large tables. No input files are executed. */
#include <assert.h>
#include <stdio.h>
#include <rz_flag.h>

int main(void) {
    RzFlag *flags = rz_flag_new();
    assert(flags);
    rz_flag_space_set(flags, "one");
    assert(rz_flag_set(flags, "first", 42, 1));
    rz_flag_space_set(flags, "two");
    RzFlagItem *alias = rz_flag_set(flags, "alias", 42, 1);
    assert(alias);
    rz_flag_item_set_alias(alias, "41+1");
    assert(rz_flag_set(flags, "neighbour", 43, 1));
    assert(rz_flag_unset_all_off(flags, 42));
    assert(!rz_flag_get(flags, "first"));
    assert(!rz_flag_get(flags, "alias"));
    assert(!rz_flag_get_list(flags, 42));
    assert(rz_flag_get(flags, "neighbour"));
    assert(rz_flag_unset_all_off(flags, 42));
    for (unsigned i = 0; i < 100000; i++) {
        char name[32];
        snprintf(name, sizeof(name), "flag_%u", i);
        assert(rz_flag_set(flags, name, 1000 + i, 1));
    }
    for (unsigned i = 0; i < 100000; i++) {
        assert(rz_flag_unset_all_off(flags, 1000 + i));
        assert(!rz_flag_get_list(flags, 1000 + i));
    }
    assert(rz_flag_get(flags, "neighbour"));
    rz_flag_free(flags);
    puts("Rizin offset-index regression passed");
    return 0;
}
